use crate::openai::{Message, OpenAIClient, OpenAIRequest, OpenAIResponse};
use crate::embedding::Embedder;
use crate::ann::{Ann, AnnResult};
use crate::ann::ChunkMeta;
use anyhow::anyhow;
use vector::Vector;
use crate::rerank::Reranker;

pub struct Hyde<'a, const D: usize> {
    pub openai: OpenAIClient,
    pub embedder: Embedder,
    pub ann: &'a Ann<D, ChunkMeta>,
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
    pub answer: String,
    pub code_refs: Vec<HydeResult>,
}

impl<'a, const D: usize> Hyde<'a, D> {
    pub fn new(openai: OpenAIClient, embedder: Embedder, ann: &'a Ann<D, ChunkMeta>, chunk_size: usize, reranker: Option<Reranker>) -> Self {
        Self { openai, embedder, ann, chunk_size, reranker }
    }

    /// Generate a hypothetical document for a given query using the OpenAI client.
    pub async fn generate_hypothetical_document(&self, query: &str) -> anyhow::Result<String> {
        let prompt = format!(
            "Generate a hypothetical Rust code snippet or document that would answer the following query as if it existed in a codebase. The generated document must fit within {} characters.\n\nQuery: {}\n\nHypothetical Document:",
            self.chunk_size, query
        );
        let system = "You are a Rust code generator. Given a query, generate a plausible Rust code snippet or document that would answer it. The output must not exceed the specified chunk size.";
        self.explain_code(&prompt, Some(system)).await
    }

    /// Retrieve the top-k nearest neighbors for a query string, with optional reranking.
    pub async fn retrieve(&self, query: &str, k: usize, use_rerank: bool) -> anyhow::Result<HydeResponse> {
        let hypothetical_document = self.generate_hypothetical_document(query).await?;
        let mut results = self.similarity_search(&hypothetical_document, k).await?;
        if use_rerank {
            if let Some(reranker) = &self.reranker {
                let docs: Vec<&str> = results.iter().map(|r| r.meta.code.as_str()).collect();
                let scores = reranker.score(vec![query], docs.clone(), 1)?;
                // Pair each result with its score
                let mut scored_results: Vec<(f32, HydeResult)> = results.into_iter().zip(scores.iter().map(|r| r.documents[0].relevance_score)).map(|(r, s)| (s, r)).collect();
                // Sort by score descending
                scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                results = scored_results.into_iter().map(|(_, r)| r).collect();
            }
        }
        let answer = self.synthesize_answer(query, &results).await?;
        Ok(HydeResponse { answer, code_refs: results })
    }

    pub async fn explain_code(&self, code: &str, system_prompt: Option<&str>) -> anyhow::Result<String> {
        let prompt = format!("Explain the following Rust code in detail:\n\n{}", code);
        let system = system_prompt.unwrap_or("You are a Rust expert. Explain the code clearly and concisely.");
        let req = OpenAIRequest {
            model: self.openai.model.clone(),
            messages: vec![
                Message { role: "system".to_string(), content: system.to_string() },
                Message { role: "user".to_string(), content: prompt },
            ],
            max_tokens: Some(512),
            temperature: Some(0.2),
        };
        let resp = self.openai.client
            .post(&self.openai.api_url)
            .bearer_auth(&self.openai.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow!("HTTP error: {e}"))?;
        if !resp.status().is_success() {
            return Err(anyhow!("OpenAI API error: {}", resp.text().await.unwrap_or_default()));
        }
        let resp: OpenAIResponse = resp.json().await?;
        let content = resp.choices.get(0)
            .and_then(|c| c.message.content.as_ref())
            .cloned()
            .unwrap_or_else(|| "No explanation returned.".to_string());
        Ok(content)
    }

    /// Retrieve the top-k nearest neighbors for a query string.
    pub async fn similarity_search(&self, query: &str, k: usize) -> anyhow::Result<Vec<HydeResult>> {
        // Only support D=512 for embedding output
        if D != 512 {
            return Err(anyhow::anyhow!("retrieve only supports D=512 (got D={})", D));
        }
        let embedding = self.embedder.embed(query).await?;
        // SAFETY: We checked D==512 above, so this cast is safe
        let embedding_ref: &Vector<D> = unsafe { &*(&embedding as *const [f32; 512] as *const [f32; D]) };
        let results = self.ann.query(embedding_ref, k as i32);
        let hyde_results = results
            .into_iter()
            .enumerate()
            .map(|(index, res)| HydeResult { index, distance: res.distance, meta: res.metadata.clone() })
            .collect();
        Ok(hyde_results)
    }

    /// Synthesize an LLM answer using the user query and top code references.
    pub async fn synthesize_answer(&self, query: &str, code_refs: &[HydeResult]) -> anyhow::Result<String> {
        let context_snippets: Vec<String> = code_refs.iter().map(|res| {
            format!("File: {}\nCode:\n{}\n", res.meta.file, res.meta.code)
        }).collect();
        let llm_prompt = format!(
            "Given the following user query:\n{}\n\nand these relevant code snippets:\n{}\n\nProvide a detailed answer, referencing the code where appropriate.",
            query,
            context_snippets.join("\n---\n")
        );
        self.explain_code(&llm_prompt, None).await
    }
} 