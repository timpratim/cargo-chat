use crate::openai::{Message, OpenAIClient, OpenAIRequest, StreamChoice};
use crate::embedding::Embedder;
use crate::ann::{ChunkMeta, DynamicAnn};
use anyhow::anyhow;
use crate::rerank::Reranker;
use std::sync::Arc;
use std::pin::Pin;
use futures_util::{Stream, StreamExt};
use async_stream::try_stream;

pub struct Hyde<'a> {
    pub openai: OpenAIClient,
    pub embedder: Arc<Embedder>,
    pub ann: &'a DynamicAnn<ChunkMeta>,
    pub chunk_size: usize,
    pub reranker: Option<Reranker>,
}

#[derive(Debug, Clone)]
pub struct HydeResult {
    pub index: usize,
    pub distance: f32,
    pub meta: ChunkMeta,
}

pub struct HydeResponse {
    pub answer_stream: Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'static>>,
    pub code_refs: Vec<HydeResult>,
}

impl<'a> Hyde<'a> {
    #[tracing::instrument(skip(openai, embedder, ann, reranker))]
    pub fn new(openai: OpenAIClient, embedder: Arc<Embedder>, ann: &'a DynamicAnn<ChunkMeta>, chunk_size: usize, reranker: Option<Reranker>) -> Self {
        Self { openai, embedder, ann, chunk_size, reranker }
    }

    #[tracing::instrument(skip(self, query))]
    pub async fn generate_hypothetical_document(&self, query: &str) -> anyhow::Result<String> {
        let prompt = format!(
            "Generate a hypothetical Rust code snippet or document that would answer the following query as if it existed in a codebase. The generated document must fit within {} characters.\n\nQuery: {}\n\nHypothetical Document:",
            self.chunk_size, query
        );
        let system = "You are a Rust code generator. Given a query, generate a plausible Rust code snippet or document that would answer it. The output must not exceed the specified chunk size.";
        
        let mut stream = self.explain_code_stream(&prompt, Some(system)).await?;
        let mut full_doc = String::new();
        while let Some(chunk_result) = stream.next().await {
            full_doc.push_str(&chunk_result?);
        }
        if full_doc.is_empty() {
            Err(anyhow!("Hypothetical document generation returned no content."))
        } else {
            Ok(full_doc)
        }
    }

    #[tracing::instrument(skip(self, query))]
    pub async fn retrieve(&self, query: &str, k: usize, use_rerank: bool) -> anyhow::Result<HydeResponse> {
        let hypothetical_document = self.generate_hypothetical_document(query).await?;
        let mut results = self.similarity_search(&hypothetical_document, k).await?;
        if use_rerank {
            if let Some(reranker) = &self.reranker {
                let docs: Vec<&str> = results.iter().map(|r| r.meta.code.as_str()).collect();
                let scores = reranker.score(vec![query], docs.clone(), 1)?;
                let mut scored_results: Vec<(f32, HydeResult)> = results.into_iter().zip(scores.iter().map(|r| r.documents[0].relevance_score)).map(|(r, s)| (s, r)).collect();
                scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                results = scored_results.into_iter().map(|(_, r)| r).collect();
            } else {
                // When reranking is requested but no reranker is available, sort by distance
                results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else {
            // When not using reranking, sort by distance to ensure proper ordering
            results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        }
        let answer_stream = self.synthesize_answer_stream(query, &results).await?;
        Ok(HydeResponse { answer_stream, code_refs: results })
    }

    #[tracing::instrument(skip(self, code, system_prompt))]
    pub async fn explain_code_stream(&self, code: &str, system_prompt: Option<&str>)
        -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'static>>>
    {
        let prompt = format!("Answer the following question about this Rust code:\n\n{}", code);
        let system = system_prompt.unwrap_or("You are a helpful Rust expert. Provide clear, concise answers. Focus on answering the question directly without unnecessary code analysis unless specifically requested.");
        
        let client = self.openai.client.clone();
        let api_url = self.openai.api_url.clone();
        let api_key = self.openai.api_key.clone();
        let model = self.openai.model.clone();

        let req = OpenAIRequest {
            model,
            messages: vec![
                Message { role: "system".to_string(), content: system.to_string() },
                Message { role: "user".to_string(), content: prompt },
            ],
            max_tokens: Some(1024),
            temperature: Some(0.2),
            stream: Some(true),
        };

        tracing::debug!("Sending OpenAI request: {:?}", req);

        let s = try_stream! {
            let http_response = client
                .post(&api_url)
                .bearer_auth(&api_key)
                .json(&req)
                .send()
                .await
                .map_err(|e| anyhow!("HTTP error sending request: {e}"))?;

            let status = http_response.status();
            tracing::debug!("Received OpenAI response status: {}", status);

            if !status.is_success() {
                let err_text = http_response.text().await.unwrap_or_else(|_| "Unknown error while reading error body".to_string());
                tracing::error!("OpenAI API error: Status {}, Body: {}", status, err_text);
                Err(anyhow!("OpenAI API error (status {}): {}", status, err_text))?;
            } else {
                let mut byte_stream = http_response.bytes_stream();
                'outer: while let Some(item) = byte_stream.next().await {
                    let chunk_bytes = item.map_err(|e| anyhow!("Error reading stream chunk bytes: {e}"))?;
                    let chunk_str = String::from_utf8_lossy(&chunk_bytes);
                    tracing::trace!("Received stream chunk (raw): <<<{}>>>", chunk_str);

                    for line_cow in chunk_str.lines() {
                        let line = line_cow.trim();
                        if line.is_empty() { continue; }
                        tracing::trace!("Processing line from chunk: '{}'", line);

                        if line.starts_with("data:") {
                            let json_data_str = line.trim_start_matches("data:").trim();
                            tracing::debug!("Extracted JSON data string: <<<{}>>>", json_data_str);

                            if json_data_str == "[DONE]" {
                                tracing::info!("SSE stream [DONE] received.");
                                break 'outer;
                            }
                            if json_data_str.is_empty() {
                                tracing::warn!("Empty JSON data after 'data:' prefix and trim.");
                                continue;
                            }

                            match serde_json::from_str::<StreamChoice>(json_data_str) {
                                Ok(stream_choice) => {
                                    tracing::debug!("Successfully parsed StreamChoice: {:?}", stream_choice);
                                    if let Some(choice) = stream_choice.choices.get(0) {
                                        if let Some(delta) = &choice.delta {
                                            if let Some(content_chunk) = &delta.content {
                                                tracing::info!("Yielding content chunk: '{}'", content_chunk);
                                                yield content_chunk.clone();
                                            }
                                        }
                                        if let Some(reason) = &choice.finish_reason {
                                            tracing::info!("Choice finish_reason received: '{}'.", reason);
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Failed to parse stream JSON data: '{}'. Error: {}. Skipping.", json_data_str, e);
                                }
                            }
                        }
                    }
                }
                tracing::info!("Finished processing OpenAI stream.");
            }
        };
        Ok(Box::pin(s))
    }

    #[tracing::instrument(skip(self, query))]
    pub async fn similarity_search(&self, query: &str, k: usize) -> anyhow::Result<Vec<HydeResult>> {
        let embedding = self.embedder.embed(query).await?;
        let results = self.ann.query(&embedding, k as i32)?;
        let hyde_results = results
            .into_iter()
            .enumerate()
            .map(|(idx, res)| HydeResult { index: idx, distance: res.distance, meta: res.metadata.clone() })
            .collect();
        Ok(hyde_results)
    }

    #[tracing::instrument(skip(self, query, code_refs))]
    pub async fn synthesize_answer_stream(&self, query: &str, code_refs: &[HydeResult])
        -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send + 'static>>>
    {
        let context_snippets: Vec<String> = code_refs.iter().map(|res| {
            format!("File: {}\nCode:\n{}\n", res.meta.file, res.meta.code)
        }).collect();
        let llm_prompt = format!(
            "User question: {}\n\nRelevant code snippets:\n{}\n\nProvide a concise, direct answer to the user's question. Only reference specific code details if they directly support your answer.",
            query,
            context_snippets.join("\n---\n")
        );
        self.explain_code_stream(&llm_prompt, None).await
    }
} 