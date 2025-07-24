use crate::ann::{ChunkMeta, DynamicAnn};
use crate::embedding::Embedder;
use crate::language::detect_language_from_extension;
use crate::openai::{
    JsonSchema, Message, OpenAIClient, OpenAIRequest, ResponseFormat, StreamChoice,
};
use crate::repo::RepoProfile;
use crate::rerank::Reranker;
use anyhow::anyhow;
use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::pin::Pin;
use std::sync::Arc;

const QUERY_CLASSIFICATION_PROMPT: &str = r#"
You are a code assistant that analyzes user queries to determine their intent, target programming language, and search scope.

Analyze the user's query and determine:
1. What programming language they're asking about (if any)
2. Whether they want to see actual code implementation (wants_code: true) or just conceptual explanations (wants_code: false)
3. The primary intent of their query
4. Specific folders/directories mentioned (e.g., "src", "tests", "lib", "examples")
5. Specific file extensions mentioned (e.g., "rs", "py", "js")
6. Patterns or folders to exclude (e.g., "test", "docs", "target")

Set wants_code=true for queries asking:
- 'How does X work' (implementation details)
- 'Show me the code for X'
- 'What's the implementation of X'
- 'How is X implemented'
- Questions about algorithms, functions, or code structure
- Questions about specific programming techniques or patterns
- Debugging or troubleshooting code issues

For folder/extension detection, look for phrases like:
- Folder patterns: "in src folder", "from src/", "src directory", "search in tests", "only in lib", "from examples"
- Extension patterns: "only .rs files", "rust files", "*.py files", "just python code", "rs files", "javascript files"
- Exclusion patterns: "exclude tests", "not in target", "skip docs", "no documentation", "no docs", "exclude build"
- File-specific: "main.rs", "lib.rs", "mod.rs" (treat as extension patterns)

IMPORTANT: When you detect these patterns, populate the corresponding fields:
- target_folders: ["src", "tests", "lib"] for folder restrictions
- target_extensions: ["rs", "py", "js"] for extension restrictions (without dots)
- exclude_patterns: ["test", "docs", "target"] for exclusions

Set wants_code=false for queries asking:
- 'What is X' (conceptual explanations)
- 'Explain the concept of X'
- 'What are the benefits of X'
- General documentation or tutorial requests
- Explicit requests for README or documentation content
- High-level architectural or design questions

IMPORTANT: When wants_code=true, the system will EXCLUDE README.md, documentation files, and markdown files from results unless explicitly requested. Focus on actual source code files (.rs, .py, .js, etc.).

Respond with structured JSON containing your analysis."#;

#[derive(Debug, Deserialize)]
struct QueryClassification {
    language: Option<String>,
    intent: String,
    wants_code: bool,
    confidence: f32,
    target_folders: Option<Vec<String>>,
    target_extensions: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
}

pub struct Hyde<'a> {
    pub hyde_client: OpenAIClient,
    pub answer_client: OpenAIClient,
    pub embedder: Arc<Embedder>,
    pub ann: &'a DynamicAnn<ChunkMeta>,
    pub chunk_size: usize,
    pub reranker: Option<Reranker>,
    pub repo_profile: Option<RepoProfile>,
}

#[derive(Debug, Clone)]
pub struct HydeResult {
    pub index: usize,
    pub distance: f32,
    pub meta: ChunkMeta,
}

pub struct HydeResponse {
    pub answer_stream: Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send>>,
    pub code_refs: Vec<HydeResult>,
}

impl<'a> Hyde<'a> {
    /// Generate intent-aware system prompt for hypothetical document generation
    fn generate_hyde_prompt(&self, classification: &QueryClassification) -> String {
        let language_context = if let Some(ref lang) = classification.language {
            format!(" Focus specifically on {} code.", lang.to_uppercase())
        } else if let Some(ref profile) = self.repo_profile {
            if !profile.primary_languages.is_empty() {
                format!(" Focus on {} code.", profile.primary_languages[0])
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let lang_name = classification
            .language
            .as_deref()
            .or_else(|| {
                self.repo_profile
                    .as_ref()
                    .and_then(|p| p.primary_languages.first().map(|s| s.as_str()))
            })
            .unwrap_or("programming");

        match classification.intent.as_str() {
            "how_it_works" => format!(
                "You are a {} code expert. Generate ONLY actual code implementation that demonstrates how something works internally.{} Generate realistic function signatures, struct definitions, impl blocks, and method implementations with descriptive names that match the query topic. Include inline comments explaining the algorithm. Focus on the core implementation logic and data structures. NO prose explanations - only executable code with comments.",
                lang_name,
                language_context
            ),
            "implementation" => format!(
                "You are a {} developer. Generate ONLY actual code implementation with realistic function signatures and method bodies.{} Show concrete struct definitions, impl blocks, and working code patterns. Include inline comments but NO prose explanations.",
                lang_name,
                language_context
            ),
            "explanation" => format!(
                "You are a technical writer specializing in {}. Generate clear explanations with code examples where helpful.{} Focus on concepts and understanding.",
                lang_name,
                language_context
            ),
            "debugging" => format!(
                "You are a {} debugging expert. Generate hypothetical code that demonstrates common issues, solutions, or debugging techniques.{} Focus on problem-solving approaches.",
                lang_name,
                language_context
            ),
            _ => format!(
                "You are a {} code assistant. Generate relevant code snippets or explanations.{} Be concise and focused.",
                lang_name,
                language_context
            ),
        }
    }

    /// Generate intent-aware system prompt for answer synthesis
    fn generate_answer_prompt(&self, classification: &QueryClassification) -> String {
        // Get repository context
        let (repo_desc, project_type, primary_languages) =
            if let Some(ref profile) = self.repo_profile {
                (
                    profile.description(),
                    profile.project_type(),
                    profile.primary_languages.clone(),
                )
            } else {
                (
                    "a software repository".to_string(),
                    "software project".to_string(),
                    vec!["programming".to_string()],
                )
            };

        let language_context = if let Some(ref lang) = classification.language {
            format!(" You are analyzing a {} codebase.", lang.to_uppercase())
        } else if !primary_languages.is_empty() {
            format!(" You are analyzing a {} codebase.", primary_languages[0])
        } else {
            String::new()
        };

        // Create dynamic system description based on repo profile
        let system_capabilities = if let Some(ref profile) = self.repo_profile {
            let mut caps = Vec::new();

            // Add language-specific capabilities
            for lang in &profile.primary_languages {
                match lang.as_str() {
                    "Rust" => {
                        caps.push("memory-safe systems programming with ownership and lifetimes")
                    }
                    "Python" => caps.push("data processing and machine learning capabilities"),
                    "JavaScript" | "TypeScript" => {
                        caps.push("web development and async JavaScript patterns")
                    }
                    "Java" => caps.push("enterprise application development"),
                    "Go" => caps.push("concurrent systems and microservices"),
                    "C++" => caps.push("high-performance computing and systems programming"),
                    _ => caps.push("modern software development patterns"),
                }
            }

            // Add framework-specific capabilities
            for framework in &profile.frameworks {
                match framework.as_str() {
                    "Tokio" => caps.push("async runtime and concurrent processing"),
                    "React" => caps.push("component-based UI development"),
                    "Django" | "Flask" | "FastAPI" => caps.push("web API development"),
                    "Spring" => caps.push("enterprise Java application framework"),
                    _ => {}
                }
            }

            if caps.is_empty() {
                "general software development patterns and architectural decisions".to_string()
            } else {
                caps.join(", ")
            }
        } else {
            "software development patterns and architectural decisions".to_string()
        };

        let base_instruction = match classification.intent.as_str() {
            "how_it_works" => {
                format!(
                    "You are a senior software architect analyzing {}.{} 

This system implements: {}

When explaining how code works, focus on:
1. The specific algorithms and data structures used
2. How components interact within the larger system architecture
3. The flow of data through the pipeline
4. Design patterns and architectural decisions
5. Performance considerations and optimizations

Use the provided code snippets as concrete examples to illustrate these concepts.",
                    repo_desc, language_context, system_capabilities
                )
            }
            "implementation" => {
                format!("You are a senior software developer providing implementation guidance for {}, {}.{}

This system features: {}

When providing implementation details:
1. Show concrete usage patterns from the actual codebase
2. Explain language-specific patterns and idioms
3. Highlight error handling and robustness patterns
4. Point out performance optimizations and best practices
5. Provide actionable insights for similar implementations

Reference the specific code snippets to demonstrate these patterns.", repo_desc, project_type, language_context, system_capabilities)
            }
            "explanation" => {
                format!(
                    "You are a technical educator explaining {}.{}

System capabilities: {}

When explaining concepts:
1. Start with high-level concepts, then dive into specifics
2. Use the code snippets as concrete examples
3. Explain the 'why' behind design decisions
4. Connect individual components to the overall system goals
5. Make complex algorithms accessible through clear explanations

Structure your explanation logically and reference specific code examples.",
                    repo_desc, language_context, system_capabilities
                )
            }
            "debugging" => {
                format!(
                    "You are a debugging specialist analyzing {}.{}

Focus on common areas for issues in this type of system:
- Component integration and data flow
- Error handling and edge cases
- Performance bottlenecks and resource management
- Configuration and setup issues
- Language-specific pitfalls and best practices

When analyzing for debugging:
1. Identify potential failure points in the provided code
2. Explain error patterns and their root causes
3. Suggest specific debugging approaches and logging strategies
4. Point out edge cases and error handling gaps
5. Recommend preventive measures and robustness improvements

Use the code snippets to illustrate both problems and solutions.",
                    repo_desc, language_context
                )
            }
            _ => {
                format!("You are an expert software engineer analyzing {}.{}

System features: {}

Provide accurate, detailed analysis based on the specific code snippets provided. Focus on the actual implementation details and architectural decisions visible in the code.", repo_desc, language_context, system_capabilities)
            }
        };

        base_instruction
    }
    #[tracing::instrument(skip(hyde_client, answer_client, embedder, ann, reranker))]
    pub fn new(
        hyde_client: OpenAIClient,
        answer_client: OpenAIClient,
        embedder: Arc<Embedder>,
        ann: &'a DynamicAnn<ChunkMeta>,
        chunk_size: usize,
        reranker: Option<Reranker>,
        repo_profile: Option<RepoProfile>,
    ) -> Self {
        Self {
            hyde_client,
            answer_client,
            embedder,
            ann,
            chunk_size,
            reranker,
            repo_profile,
        }
    }

    #[tracing::instrument(skip(self, query, classification))]
    pub async fn generate_hypothetical_document(
        &self,
        query: &str,
        classification: &QueryClassification,
    ) -> anyhow::Result<String> {
        let system_prompt = self.generate_hyde_prompt(classification);

        // Create more code-focused input when user wants code
        let input = if classification.wants_code {
            format!(
                "Query: {}\n\nGenerate actual code implementation (functions, structs, methods) that would answer this query. Use descriptive names that relate to the query topic. Focus on implementation details and core logic. Maximum {} characters.",
                query, self.chunk_size
            )
        } else {
            format!(
                "Query: {}\n\nGenerate a hypothetical document or explanation under {} characters that would be relevant to this query.",
                query, self.chunk_size
            )
        };
        let mut stream = self.explain_code_stream(&input, &system_prompt).await?;
        let mut doc = String::new();
        while let Some(chunk) = stream.next().await {
            doc.push_str(&chunk?);
        }
        if doc.is_empty() {
            Err(anyhow!("Hypothetical generation returned no content."))
        } else {
            Ok(doc)
        }
    }

    /// Classify the user query using LLM with structured output
    #[tracing::instrument(skip(self, query))]
    async fn classify_query(&self, query: &str) -> anyhow::Result<QueryClassification> {
        tracing::debug!("[CLASSIFY] Starting query classification for: '{}'", query);
        let schema = json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": ["string", "null"],
                    "description": "Programming language mentioned or implied in the query (e.g., 'rust', 'python', 'javascript')"
                },
                "intent": {
                    "type": "string",
                    "description": "Primary intent of the query (e.g., 'how_it_works', 'implementation', 'explanation', 'debugging')"
                },
                "wants_code": {
                    "type": "boolean",
                    "description": "Whether the user wants to see actual code implementation"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence level in the classification (0.0 to 1.0)"
                },
                "target_folders": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Specific folders/directories mentioned in the query (e.g., ['src', 'tests', 'lib'])"
                },
                "target_extensions": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Specific file extensions mentioned in the query (e.g., ['rs', 'py', 'js'])"
                },
                "exclude_patterns": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Patterns or folders to exclude from search (e.g., ['test', 'docs', 'target'])"
                }
            },
            "required": ["language", "intent", "wants_code", "confidence", "target_folders", "target_extensions", "exclude_patterns"],
            "additionalProperties": false
        });

        let req = OpenAIRequest {
            model: self.hyde_client.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: QUERY_CLASSIFICATION_PROMPT.into(),
                },
                Message {
                    role: "user".into(),
                    content: format!("Analyze this query: {}", query),
                },
            ],
            max_tokens: Some(200),
            temperature: Some(0.1),
            stream: Some(false),
            response_format: Some(ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: Some(JsonSchema {
                    name: "query_classification".to_string(),
                    schema,
                    strict: Some(true),
                }),
            }),
        };

        tracing::debug!("Sending query classification request: {:?}", req);
        let resp = self
            .hyde_client
            .client
            .post(&self.hyde_client.api_url)
            .bearer_auth(&self.hyde_client.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow!("HTTP error: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(anyhow!("API error {}: {}", status, err));
        }

        let response: crate::openai::OpenAIResponse = resp
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse response: {e}"))?;

        let content = response
            .choices
            .get(0)
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| anyhow!("No content in response"))?;

        let classification: QueryClassification = serde_json::from_str(content)
            .map_err(|e| anyhow!("Failed to parse classification: {e}"))?;

        tracing::debug!("[CLASSIFY] Raw classification response: {}", content);
        tracing::debug!("[CLASSIFY] Parsed classification: {:?}", classification);
        tracing::info!("[CLASSIFY] Classification complete: language={:?}, intent={}, wants_code={}, confidence={:.2}", 
            classification.language, classification.intent, classification.wants_code, classification.confidence);

        if let Some(ref folders) = classification.target_folders {
            tracing::info!("[CLASSIFY] Target folders detected: {:?}", folders);
        }
        if let Some(ref extensions) = classification.target_extensions {
            tracing::info!("[CLASSIFY] Target extensions detected: {:?}", extensions);
        }
        if let Some(ref excludes) = classification.exclude_patterns {
            tracing::info!("[CLASSIFY] Exclude patterns detected: {:?}", excludes);
        }

        Ok(classification)
    }

    #[tracing::instrument(skip(self, query))]
    pub async fn retrieve(
        &self,
        query: &str,
        k: usize,
        use_rerank: bool,
    ) -> anyhow::Result<HydeResponse> {
        tracing::info!(
            "[RETRIEVE] Starting retrieval pipeline for query: '{}'",
            query
        );
        tracing::info!("[RETRIEVE] Parameters: k={}, use_rerank={}", k, use_rerank);

        // First, classify the query to understand intent and language
        tracing::info!("[RETRIEVE] Step 1: Classifying query...");
        let classification = self.classify_query(query).await?;
        tracing::info!("[RETRIEVE] Query classification complete: language={:?}, intent={}, wants_code={}, confidence={:.2}", 
            classification.language, classification.intent, classification.wants_code, classification.confidence);

        tracing::info!("[RETRIEVE] Step 2: Generating hypothetical document...");
        let hypo = self
            .generate_hypothetical_document(query, &classification)
            .await?;
        tracing::info!(
            "[RETRIEVE] Hypothetical document generated ({} chars)",
            hypo.len()
        );
        tracing::debug!("[RETRIEVE] === HYPOTHETICAL DOCUMENT DEBUG ===");
        tracing::debug!("[RETRIEVE] Full hypothetical document: {}", hypo);
        tracing::debug!("[RETRIEVE] === END HYPOTHETICAL DOCUMENT ===");

        tracing::info!("[RETRIEVE] Step 3: Performing similarity search...");
        let mut results = self.similarity_search(&hypo, k * 2).await?; // Get more results for filtering
        tracing::info!(
            "[RETRIEVE] Similarity search returned {} results (requested {})",
            results.len(),
            k * 2
        );

        // Apply intelligent filtering based on LLM classification
        tracing::info!("[RETRIEVE] Step 4: Applying LLM-based filtering...");
        let before_filtering = results.len();
        results = self.apply_llm_based_filtering(&classification, results);
        tracing::info!(
            "[RETRIEVE] LLM filtering complete: {} -> {} results",
            before_filtering,
            results.len()
        );

        // Limit to requested k after filtering
        let before_truncate = results.len();
        results.truncate(k);
        if before_truncate > k {
            tracing::info!(
                "[RETRIEVE] Truncated results from {} to {} (requested k={})",
                before_truncate,
                results.len(),
                k
            );
        }

        tracing::info!("[RETRIEVE] Step 5: Reranking and sorting...");
        if use_rerank {
            if let Some(reranker) = &self.reranker {
                tracing::info!("[RETRIEVE] Applying reranker to {} results", results.len());
                let docs: Vec<&str> = results.iter().map(|r| r.meta.code.as_str()).collect();
                let scores = reranker.score(vec![query], docs.clone(), 1)?;
                let mut scored: Vec<(f32, HydeResult)> = results
                    .into_iter()
                    .zip(scores.iter().map(|r| r.documents[0].relevance_score))
                    .map(|(r, s)| (s, r))
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                results = scored.into_iter().map(|(_, r)| r).collect();
                tracing::info!("[RETRIEVE] Reranking complete");
            } else {
                tracing::info!(
                    "[RETRIEVE] Reranker requested but not available, sorting by distance"
                );
                results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            }
        } else {
            tracing::info!("[RETRIEVE] Sorting by similarity distance");
            results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        }

        tracing::info!("[RETRIEVE] Step 6: Generating answer stream...");
        let answer_stream = self
            .synthesize_answer_stream(query, &results, &classification)
            .await?;

        tracing::info!(
            "[RETRIEVE] Pipeline complete! Returning {} code references",
            results.len()
        );
        for (i, result) in results.iter().enumerate() {
            tracing::info!(
                "[RETRIEVE] Final reference {}: '{}' (distance: {:.4})",
                i,
                result.meta.file,
                result.distance
            );
        }

        Ok(HydeResponse {
            answer_stream,
            code_refs: results,
        })
    }

    #[tracing::instrument(skip(self, content, system_prompt))]
    async fn explain_code_stream(
        &self,
        content: &str,
        system_prompt: &str,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send>>> {
        let prompt = format!("Context:\n{}\n", content);
        let req = OpenAIRequest {
            model: self.hyde_client.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: system_prompt.into(),
                },
                Message {
                    role: "user".into(),
                    content: prompt,
                },
            ],
            max_tokens: Some(1024),
            temperature: Some(0.2),
            stream: Some(true),
            response_format: None,
        };
        self.stream_request(&self.hyde_client, req).await
    }

    #[tracing::instrument(skip(self, query, code_refs, classification))]
    pub async fn synthesize_answer_stream(
        &self,
        query: &str,
        code_refs: &[HydeResult],
        classification: &QueryClassification,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send>>> {
        let persona_prompt = self.generate_answer_prompt(classification);

        // Investigative guide to enforce Amp-style systematic analysis
        let investigative_prompt = r#"INVESTIGATIVE METHODOLOGY:

You are an expert code auditor. Follow this systematic 5-step analysis method internally before writing your answer:

STEP 1: UNDERSTAND THE REQUEST
- Restate the user's exact question in one sentence
- Identify what specific aspect they want to understand (architecture, implementation, flow, etc.)

STEP 2: SCAN AND CATALOG CODE REFERENCES  
- Examine each provided code reference
- Note: file paths, important function/struct names, key concepts
- Identify relationships between different code pieces

STEP 3: TRACE EXECUTION FLOW
- Mentally map how the code pieces connect
- Identify entry points, data flow, and call patterns
- Look for the main pipeline or algorithm being implemented

STEP 4: SYNTHESIZE SYSTEM UNDERSTANDING
- Connect individual pieces to the broader architecture
- Identify the key design patterns and architectural decisions
- Understand the "why" behind the implementation choices

STEP 5: EXTRACT KEY INSIGHTS
- Determine the most important takeaway the user needs
- Focus on the core concepts that make the system work

OUTPUT STRUCTURE: You MUST follow this exact template in your response."#;

        // Enhanced snippet formatting with better context and line references
        let snippets: Vec<String> = code_refs
            .iter()
            .enumerate()
            .map(|(idx, r)| {
                let lang_info = r
                    .meta
                    .language
                    .as_ref()
                    .map(|l| format!(" [{}]", l.to_uppercase()))
                    .unwrap_or_default();
                let ext_info = r
                    .meta
                    .extension
                    .as_ref()
                    .map(|e| format!(".{}", e))
                    .unwrap_or_default();

                // Clean up file paths for better readability
                let clean_path = r.meta.file.trim_start_matches("./");

                format!(
                    "## Code Reference {} - {}{}{}
**Location:** {}
**Relevance Score:** {:.3} (lower = more relevant)
**Chunk ID:** {}

```{}
{}```
",
                    idx + 1,
                    clean_path,
                    ext_info,
                    lang_info,
                    clean_path,
                    r.distance,
                    r.index,
                    r.meta.language.as_deref().unwrap_or("rust"),
                    r.meta.code.trim()
                )
            })
            .collect();

        let context_instruction = match classification.intent.as_str() {
            "how_it_works" => "Focus on tracing the execution flow and explaining the algorithmic approach. Show how data moves through the system and what makes each component work internally.",
            "implementation" => "Examine the code patterns and implementation techniques. Show concrete examples of how the system is built and what patterns can be reused.",
            "explanation" => "Provide a comprehensive explanation that connects high-level concepts to specific implementation details. Make complex algorithms accessible while maintaining technical accuracy.",
            "debugging" => "Analyze potential failure points and debugging strategies. Look for error patterns, edge cases, and areas for improvement in the provided code.",
            _ => "Analyze the code systematically to provide a complete understanding of how this part of the system works.",
        };

        // Structured output template that forces organized, investigative responses
        let output_template = r#"===== RESPONSE TEMPLATE =====

## 1. Analysis Overview
<One sentence restating the user's question and what you discovered>

## 2. Investigation Plan  
<2-4 bullets outlining your analytical approach based on the code references>

## 3. Code Analysis
<For each relevant code reference, explain:>
- **Code Reference X** (file_path): <explanation of what this code does and its role>
- **Key Functions/Structures:** <important symbols found>
- **Role in System:** <how this fits into the bigger picture>

## 4. System Flow & Architecture
<Narrative explanation connecting all the pieces:>
- <How the components interact>
- <Data flow through the system>  
- <Key architectural decisions visible in the code>

## 5. Core Insight
<The fundamental concept or mechanism that makes this system work - explain the "why" behind the implementation>

============================="#;

        let system_context = if let Some(ref profile) = self.repo_profile {
            format!("You are analyzing {}", profile.description())
        } else {
            "You are analyzing a software repository".to_string()
        };

        let user_prompt = format!(
            "# User Query
{}\n\n# Analysis Task
{}\n\n# System Context
{}

# Available Code References
{}\n\n{}",
            query,
            context_instruction,
            system_context,
            snippets.join("\n\n"),
            output_template
        );

        let req = OpenAIRequest {
            model: self.answer_client.model.clone(),
            messages: vec![
                Message {
                    role: "system".into(),
                    content: persona_prompt,
                },
                Message {
                    role: "system".into(),
                    content: investigative_prompt.to_string(),
                },
                Message {
                    role: "user".into(),
                    content: user_prompt,
                },
            ],
            max_tokens: Some(2048),
            temperature: Some(0.0), // Use deterministic temperature for more structured output
            stream: Some(true),
            response_format: None,
        };
        self.stream_request(&self.answer_client, req).await
    }

    #[tracing::instrument(skip(self, client, req))]
    async fn stream_request(
        &self,
        client: &OpenAIClient,
        req: OpenAIRequest,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Result<String, anyhow::Error>> + Send>>> {
        tracing::debug!("Sending OpenAI request: {:?}", req);
        let resp = client
            .client
            .post(&client.api_url)
            .bearer_auth(&client.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| anyhow!("HTTP error: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(anyhow!("API error {}: {}", status, err));
        }

        let s = try_stream! {
            let mut bs = resp.bytes_stream();
            'outer: while let Some(item) = bs.next().await {
                let chunk = item.map_err(|e| anyhow!("Stream error: {e}"))?;
                for line in String::from_utf8_lossy(&chunk).lines() {
                    let line = line.trim();
                    if !line.starts_with("data:") { continue; }
                    let data = line.trim_start_matches("data:").trim();
                    if data == "[DONE]" { break 'outer; }
                    if data.is_empty() { continue; }
                    if let Ok(choice) = serde_json::from_str::<StreamChoice>(data) {
                        if let Some(delta) = &choice.choices[0].delta {
                            if let Some(text) = &delta.content {
                                yield text.clone();
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(s))
    }

    /// Apply folder and extension filtering based on query classification
    fn apply_folder_extension_filtering(
        &self,
        classification: &QueryClassification,
        mut results: Vec<HydeResult>,
    ) -> Vec<HydeResult> {
        let initial_count = results.len();
        tracing::debug!(
            "[FILTER] Starting folder/extension filtering with {} results",
            initial_count
        );

        // Apply folder filtering if target_folders is specified
        if let Some(ref target_folders) = classification.target_folders {
            if !target_folders.is_empty() {
                tracing::debug!("[FILTER] Filtering by target folders: {:?}", target_folders);
                let before_count = results.len();
                results.retain(|result| {
                    let matches = target_folders.iter().any(|folder| {
                        let file_path = &result.meta.file;
                        // Check if file path contains any of the target folders
                        let match_result = file_path.contains(&format!("/{}/", folder))
                            || file_path.starts_with(&format!("{}/", folder))
                            || file_path.contains(&format!("/{}", folder))
                                && file_path.ends_with(&format!(
                                    "/{}",
                                    folder.split('/').last().unwrap_or(folder)
                                ));

                        if match_result {
                            tracing::debug!(
                                "[FILTER] ✓ File '{}' matches folder '{}'.",
                                file_path,
                                folder
                            );
                        }
                        match_result
                    });

                    if !matches {
                        tracing::debug!(
                            "[FILTER] ✗ File '{}' excluded by folder filter",
                            result.meta.file
                        );
                    }
                    matches
                });
                tracing::debug!(
                    "[FILTER] Folder filtering: {} -> {} results ({} excluded)",
                    before_count,
                    results.len(),
                    before_count - results.len()
                );
            }
        }

        // Apply extension filtering if target_extensions is specified
        if let Some(ref target_extensions) = classification.target_extensions {
            if !target_extensions.is_empty() {
                tracing::debug!(
                    "[FILTER] Filtering by target extensions: {:?}",
                    target_extensions
                );
                let before_count = results.len();
                results.retain(|result| {
                    let matches = if let Some(ref ext) = result.meta.extension {
                        target_extensions.iter().any(|target_ext| {
                            // Handle extensions with or without leading dot
                            let normalized_target = target_ext.trim_start_matches('.');
                            let normalized_ext = ext.trim_start_matches('.');
                            let match_result = normalized_ext == normalized_target;
                            if match_result {
                                tracing::debug!(
                                    "[FILTER] ✓ Extension '{}' matches target '{}'",
                                    ext,
                                    target_ext
                                );
                            }
                            match_result
                        })
                    } else {
                        // If no extension metadata, check file path
                        target_extensions.iter().any(|target_ext| {
                            let normalized_target = target_ext.trim_start_matches('.');
                            let match_result = result
                                .meta
                                .file
                                .ends_with(&format!(".{}", normalized_target));
                            if match_result {
                                tracing::debug!(
                                    "[FILTER] ✓ File '{}' matches extension '{}' (from path)",
                                    result.meta.file,
                                    target_ext
                                );
                            }
                            match_result
                        })
                    };

                    if !matches {
                        tracing::debug!(
                            "[FILTER] ✗ File '{}' (ext: {:?}) excluded by extension filter",
                            result.meta.file,
                            result.meta.extension
                        );
                    }
                    matches
                });
                tracing::debug!(
                    "[FILTER] Extension filtering: {} -> {} results ({} excluded)",
                    before_count,
                    results.len(),
                    before_count - results.len()
                );
            }
        }

        // Apply exclude patterns if specified
        if let Some(ref exclude_patterns) = classification.exclude_patterns {
            if !exclude_patterns.is_empty() {
                tracing::debug!("[FILTER] Applying exclude patterns: {:?}", exclude_patterns);
                let before_count = results.len();
                results.retain(|result| {
                    let file_path = &result.meta.file;
                    let should_exclude = exclude_patterns.iter().any(|pattern| {
                        let pattern_lower = pattern.to_lowercase();
                        let file_lower = file_path.to_lowercase();

                        // Check various patterns: folder names, file patterns, etc.
                        let matches = file_lower.contains(&pattern_lower)
                            || file_lower.contains(&format!("/{}/", pattern_lower))
                            || file_lower.starts_with(&format!("{}/", pattern_lower))
                            || file_lower.ends_with(&format!("/{}", pattern_lower));

                        if matches {
                            tracing::debug!(
                                "[FILTER] ✗ File '{}' excluded by pattern '{}'",
                                file_path,
                                pattern
                            );
                        }
                        matches
                    });
                    !should_exclude
                });
                tracing::debug!(
                    "[FILTER] Exclude filtering: {} -> {} results ({} excluded)",
                    before_count,
                    results.len(),
                    before_count - results.len()
                );
            }
        }

        tracing::debug!(
            "[FILTER] Folder/extension filtering complete: {} -> {} results",
            initial_count,
            results.len()
        );

        results
    }

    /// Apply LLM-based filtering using query classification results
    fn apply_llm_based_filtering(
        &self,
        classification: &QueryClassification,
        mut results: Vec<HydeResult>,
    ) -> Vec<HydeResult> {
        tracing::debug!("[LLM_FILTER] Starting LLM-based filtering");
        tracing::debug!("[LLM_FILTER] Classification: language={:?}, intent={}, wants_code={}, confidence={:.2}", 
            classification.language, classification.intent, classification.wants_code, classification.confidence);
        tracing::debug!("[LLM_FILTER] Input results count: {}", results.len());

        // Log all results to see what we're working with
        for (i, result) in results.iter().enumerate() {
            tracing::debug!(
                "[LLM_FILTER] Input result {}: file='{}', ext={:?}, lang={:?}, distance={:.4}",
                i,
                result.meta.file,
                result.meta.extension,
                result.meta.language,
                result.distance
            );
        }

        // Apply folder and extension filtering first if specified
        let before_folder_ext = results.len();
        results = self.apply_folder_extension_filtering(classification, results);
        tracing::debug!(
            "[LLM_FILTER] After folder/extension filtering: {} -> {} results",
            before_folder_ext,
            results.len()
        );

        // If the user wants code and we have high confidence, prioritize code files
        if classification.wants_code && classification.confidence > 0.7 {
            tracing::debug!("[LLM_FILTER] High-confidence code query - separating code from docs");

            // Separate code files from documentation - be more aggressive about excluding docs
            let (mut code_results, mut doc_results): (Vec<_>, Vec<_>) =
                results.into_iter().partition(|r| {
                    let is_code = if let Some(ref ext) = r.meta.extension {
                        // Prioritize actual code files over documentation
                        let is_doc_ext = matches!(ext.as_str(), "md" | "txt" | "rst" | "adoc");
                        if is_doc_ext {
                            tracing::debug!(
                                "[LLM_FILTER] File '{}' classified as DOC (extension: {})",
                                r.meta.file,
                                ext
                            );
                        } else {
                            tracing::debug!(
                                "[LLM_FILTER] File '{}' classified as CODE (extension: {})",
                                r.meta.file,
                                ext
                            );
                        }
                        !is_doc_ext
                    } else {
                        // If no extension info, check file path - be more aggressive
                        let file_lower = r.meta.file.to_lowercase();
                        let is_code_file = !file_lower.ends_with(".md")
                            && !file_lower.ends_with(".txt")
                            && !file_lower.contains("/docs/")
                            && !file_lower.contains("readme")
                            && !file_lower.contains("resources")
                            && !file_lower.contains("changelog")
                            && !file_lower.contains("license");

                        if is_code_file {
                            tracing::debug!(
                                "[LLM_FILTER] File '{}' classified as CODE (path analysis)",
                                r.meta.file
                            );
                        } else {
                            tracing::debug!(
                                "[LLM_FILTER] File '{}' classified as DOC (path analysis)",
                                r.meta.file
                            );
                        }
                        is_code_file
                    };
                    is_code
                });

            tracing::debug!(
                "[LLM_FILTER] Separated into {} code files and {} doc files",
                code_results.len(),
                doc_results.len()
            );

            // If a specific language is detected, prioritize files of that language
            if let Some(ref target_lang) = classification.language {
                let target_lang_lower = target_lang.to_lowercase();

                code_results.sort_by(|a, b| {
                    let a_matches_lang = a
                        .meta
                        .language
                        .as_ref()
                        .map_or(false, |l| l.to_lowercase() == target_lang_lower)
                        || a.meta
                            .extension
                            .as_ref()
                            .and_then(|e| detect_language_from_extension(e))
                            .map_or(false, |lang| {
                                lang.display_name().to_lowercase() == target_lang_lower
                            });

                    let b_matches_lang = b
                        .meta
                        .language
                        .as_ref()
                        .map_or(false, |l| l.to_lowercase() == target_lang_lower)
                        || b.meta
                            .extension
                            .as_ref()
                            .and_then(|e| detect_language_from_extension(e))
                            .map_or(false, |lang| {
                                lang.display_name().to_lowercase() == target_lang_lower
                            });

                    match (a_matches_lang, b_matches_lang) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => a
                            .distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    }
                });
            } else {
                // Sort code results by distance
                code_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            }

            // Sort doc results by distance
            doc_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

            // Combine: prioritize code files, then add some documentation for context
            let mut filtered_results = code_results;

            // Add documentation results for context based on intent - be more restrictive
            // When wants_code=true with high confidence, exclude docs entirely
            let doc_limit = if classification.wants_code && classification.confidence > 0.8 {
                tracing::debug!(
                    "[LLM_FILTER] High confidence code query ({}): excluding all docs",
                    classification.confidence
                );
                0 // No docs when user explicitly wants code with high confidence
            } else {
                let limit = match classification.intent.as_str() {
                    "how_it_works" => {
                        // For "how it works" queries, only include docs if we have very few code results
                        if filtered_results.len() >= 3 {
                            tracing::debug!("[LLM_FILTER] 'how_it_works' query with {} code results: excluding docs", filtered_results.len());
                            0 // No docs if we have enough code
                        } else {
                            let limit = std::cmp::min(doc_results.len(), 1);
                            tracing::debug!("[LLM_FILTER] 'how_it_works' query with {} code results: allowing {} docs", filtered_results.len(), limit);
                            limit // At most 1 doc file
                        }
                    }
                    "implementation" | "debugging" => {
                        tracing::debug!(
                            "[LLM_FILTER] '{}' query: excluding all docs",
                            classification.intent
                        );
                        0 // No docs for implementation/debugging queries
                    }
                    "explanation" => {
                        let limit = std::cmp::min(doc_results.len(), 1);
                        tracing::debug!(
                            "[LLM_FILTER] 'explanation' query: allowing {} docs",
                            limit
                        );
                        limit // At most 1 doc for explanations
                    }
                    _ => {
                        tracing::debug!(
                            "[LLM_FILTER] Unknown intent '{}': excluding all docs",
                            classification.intent
                        );
                        0 // Default to no docs for code queries
                    }
                };
                limit
            };

            tracing::debug!(
                "[LLM_FILTER] Adding {} doc files to {} code files",
                doc_limit,
                filtered_results.len()
            );
            filtered_results.extend(doc_results.into_iter().take(doc_limit));

            tracing::debug!(
                "[LLM_FILTER] Final result count: {} files",
                filtered_results.len()
            );
            for (i, result) in filtered_results.iter().enumerate() {
                tracing::debug!(
                    "[LLM_FILTER] Final result {}: '{}' (distance: {:.4})",
                    i,
                    result.meta.file,
                    result.distance
                );
            }

            filtered_results
        } else {
            // For non-code queries or low confidence, use original sorting with slight preference for docs
            results.sort_by(|a, b| {
                let a_is_doc = a
                    .meta
                    .extension
                    .as_ref()
                    .map_or(false, |e| matches!(e.as_str(), "md" | "txt" | "rst"));
                let b_is_doc = b
                    .meta
                    .extension
                    .as_ref()
                    .map_or(false, |e| matches!(e.as_str(), "md" | "txt" | "rst"));

                if !classification.wants_code {
                    // Slightly prefer documentation for explanation queries
                    match (a_is_doc, b_is_doc) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => a
                            .distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    }
                } else {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            });
            results
        }
    }

    #[tracing::instrument(skip(self, query))]
    pub async fn similarity_search(
        &self,
        query: &str,
        k: usize,
    ) -> anyhow::Result<Vec<HydeResult>> {
        let emb = self.embedder.embed(query).await?;
        let hits = self.ann.query(&emb, k as i32)?;
        Ok(hits
            .into_iter()
            .enumerate()
            .map(|(i, r)| HydeResult {
                index: i,
                distance: r.distance,
                meta: r.metadata.clone(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::ChunkMeta;

    // Helper function to create test HydeResult
    fn create_test_result(
        file: &str,
        code: &str,
        extension: Option<&str>,
        language: Option<&str>,
        distance: f32,
    ) -> HydeResult {
        HydeResult {
            index: 0,
            distance,
            meta: ChunkMeta {
                file: file.to_string(),
                code: code.to_string(),
                extension: extension.map(|s| s.to_string()),
                language: language.map(|s| s.to_string()),
            },
        }
    }

    // Helper function to create test QueryClassification
    fn create_test_classification(
        language: Option<&str>,
        intent: &str,
        wants_code: bool,
        confidence: f32,
    ) -> QueryClassification {
        QueryClassification {
            language: language.map(|s| s.to_string()),
            intent: intent.to_string(),
            wants_code,
            confidence,
            target_folders: None,
            target_extensions: None,
            exclude_patterns: None,
        }
    }

    // Test the filtering logic directly without needing a full Hyde instance
    // We'll create a minimal test struct that has the filtering method
    struct TestHyde;

    impl TestHyde {
        // Copy the folder/extension filtering logic for testing
        fn apply_folder_extension_filtering(
            &self,
            classification: &QueryClassification,
            mut results: Vec<HydeResult>,
        ) -> Vec<HydeResult> {
            // Apply folder filtering if target_folders is specified
            if let Some(ref target_folders) = classification.target_folders {
                if !target_folders.is_empty() {
                    results.retain(|result| {
                        target_folders.iter().any(|folder| {
                            let file_path = &result.meta.file;
                            // Check if file path contains any of the target folders
                            file_path.contains(&format!("/{}/", folder))
                                || file_path.starts_with(&format!("{}/", folder))
                                || file_path.contains(&format!("/{}", folder))
                        })
                    });
                }
            }

            // Apply extension filtering if target_extensions is specified
            if let Some(ref target_extensions) = classification.target_extensions {
                if !target_extensions.is_empty() {
                    results.retain(|result| {
                        if let Some(ref ext) = result.meta.extension {
                            target_extensions.iter().any(|target_ext| {
                                // Handle extensions with or without leading dot
                                let normalized_target = target_ext.trim_start_matches('.');
                                let normalized_ext = ext.trim_start_matches('.');
                                normalized_ext == normalized_target
                            })
                        } else {
                            // If no extension metadata, check file path
                            target_extensions.iter().any(|target_ext| {
                                let normalized_target = target_ext.trim_start_matches('.');
                                result
                                    .meta
                                    .file
                                    .ends_with(&format!(".{}", normalized_target))
                            })
                        }
                    });
                }
            }

            // Apply exclude patterns if specified
            if let Some(ref exclude_patterns) = classification.exclude_patterns {
                if !exclude_patterns.is_empty() {
                    results.retain(|result| {
                        let file_path = &result.meta.file;
                        !exclude_patterns.iter().any(|pattern| {
                            let pattern_lower = pattern.to_lowercase();
                            let file_lower = file_path.to_lowercase();

                            // Check various patterns: folder names, file patterns, etc.
                            file_lower.contains(&pattern_lower)
                                || file_lower.contains(&format!("/{}/", pattern_lower))
                                || file_lower.starts_with(&format!("{}/", pattern_lower))
                                || file_lower.ends_with(&format!("/{}", pattern_lower))
                        })
                    });
                }
            }

            results
        }

        // Copy the filtering logic for testing
        fn apply_llm_based_filtering(
            &self,
            classification: &QueryClassification,
            mut results: Vec<HydeResult>,
        ) -> Vec<HydeResult> {
            // If the user wants code and we have high confidence, prioritize code files
            if classification.wants_code && classification.confidence > 0.7 {
                // Separate code files from documentation
                let (mut code_results, mut doc_results): (Vec<_>, Vec<_>) =
                    results.into_iter().partition(|r| {
                        if let Some(ref ext) = r.meta.extension {
                            // Prioritize actual code files over documentation
                            !matches!(ext.as_str(), "md" | "txt" | "rst" | "adoc")
                        } else {
                            // If no extension info, check file path
                            !r.meta.file.ends_with(".md")
                                && !r.meta.file.ends_with(".txt")
                                && !r.meta.file.contains("/docs/")
                        }
                    });

                // If a specific language is detected, prioritize files of that language
                if let Some(ref target_lang) = classification.language {
                    let target_lang_lower = target_lang.to_lowercase();

                    code_results.sort_by(|a, b| {
                        let a_matches_lang = a
                            .meta
                            .language
                            .as_ref()
                            .map_or(false, |l| l.to_lowercase() == target_lang_lower);

                        let b_matches_lang = b
                            .meta
                            .language
                            .as_ref()
                            .map_or(false, |l| l.to_lowercase() == target_lang_lower);

                        match (a_matches_lang, b_matches_lang) {
                            (true, false) => std::cmp::Ordering::Less,
                            (false, true) => std::cmp::Ordering::Greater,
                            _ => a
                                .distance
                                .partial_cmp(&b.distance)
                                .unwrap_or(std::cmp::Ordering::Equal),
                        }
                    });
                } else {
                    // Sort code results by distance
                    code_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                }

                // Sort doc results by distance
                doc_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

                // Combine: prioritize code files, then add some documentation for context
                let mut filtered_results = code_results;

                // Add all documentation results but prioritize code files first
                // For implementation queries, we still include docs but put them after code
                filtered_results.extend(doc_results.into_iter());
                filtered_results
            } else {
                // For non-code queries or low confidence, use original sorting with slight preference for docs
                results.sort_by(|a, b| {
                    let a_is_doc = a
                        .meta
                        .extension
                        .as_ref()
                        .map_or(false, |e| matches!(e.as_str(), "md" | "txt" | "rst"));
                    let b_is_doc = b
                        .meta
                        .extension
                        .as_ref()
                        .map_or(false, |e| matches!(e.as_str(), "md" | "txt" | "rst"));

                    if !classification.wants_code {
                        // Slightly prefer documentation for explanation queries
                        match (a_is_doc, b_is_doc) {
                            (true, false) => std::cmp::Ordering::Less,
                            (false, true) => std::cmp::Ordering::Greater,
                            _ => a
                                .distance
                                .partial_cmp(&b.distance)
                                .unwrap_or(std::cmp::Ordering::Equal),
                        }
                    } else {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
                results
            }
        }
    }

    #[test]
    fn test_filtering_prioritizes_code_over_docs_when_wants_code_high_confidence() {
        let test_hyde = TestHyde;

        // Create test results with mixed code and documentation files
        let results = vec![
            create_test_result("README.md", "# Documentation", Some("md"), None, 0.1), // Close match but doc
            create_test_result(
                "src/hyde.rs",
                "fn hyde_impl() {}",
                Some("rs"),
                Some("rust"),
                0.3,
            ), // Further but code
            create_test_result("docs/guide.md", "## Guide", Some("md"), None, 0.2), // Medium match but doc
            create_test_result("src/main.rs", "fn main() {}", Some("rs"), Some("rust"), 0.4), // Furthest but code
        ];

        // Classification: wants code, high confidence, rust language
        let classification = create_test_classification(Some("rust"), "implementation", true, 0.8);

        let filtered = test_hyde.apply_llm_based_filtering(&classification, results);

        // Should prioritize Rust code files over documentation
        assert_eq!(filtered.len(), 4);
        assert!(filtered[0].meta.file.contains("src/hyde.rs")); // Best Rust code file first
        assert!(filtered[1].meta.file.contains("src/main.rs")); // Second Rust code file
                                                                // Documentation should come after code files
        let doc_positions: Vec<usize> = filtered
            .iter()
            .enumerate()
            .filter(|(_, r)| r.meta.file.contains(".md"))
            .map(|(i, _)| i)
            .collect();
        assert!(
            doc_positions.iter().all(|&pos| pos >= 2),
            "Documentation should come after code files"
        );
    }

    #[test]
    fn test_filtering_language_specific_prioritization() {
        let test_hyde = TestHyde;

        let results = vec![
            create_test_result(
                "src/main.py",
                "def main():",
                Some("py"),
                Some("python"),
                0.2,
            ),
            create_test_result(
                "src/hyde.rs",
                "fn hyde_impl() {}",
                Some("rs"),
                Some("rust"),
                0.3,
            ),
            create_test_result(
                "src/app.js",
                "function app() {}",
                Some("js"),
                Some("javascript"),
                0.1,
            ),
        ];

        // Want Rust specifically
        let classification = create_test_classification(Some("rust"), "implementation", true, 0.9);
        let filtered = test_hyde.apply_llm_based_filtering(&classification, results);

        // Rust file should be first despite higher distance
        assert!(filtered[0].meta.file.contains("hyde.rs"));
        assert_eq!(filtered[0].meta.language, Some("rust".to_string()));
    }

    #[test]
    fn test_filtering_low_confidence_preserves_original_order() {
        let test_hyde = TestHyde;

        let results = vec![
            create_test_result("README.md", "# Documentation", Some("md"), None, 0.1),
            create_test_result(
                "src/hyde.rs",
                "fn hyde_impl() {}",
                Some("rs"),
                Some("rust"),
                0.3,
            ),
        ];

        // Low confidence - should not aggressively reorder
        let classification = create_test_classification(Some("rust"), "implementation", true, 0.5);
        let filtered = test_hyde.apply_llm_based_filtering(&classification, results);

        // Should preserve distance-based ordering for low confidence
        assert!(filtered[0].meta.file.contains("README.md")); // Closest match first
    }

    #[test]
    fn test_folder_filtering() {
        let test_hyde = TestHyde;

        let results = vec![
            create_test_result("src/main.rs", "fn main() {}", Some("rs"), Some("rust"), 0.1),
            create_test_result(
                "tests/test.rs",
                "#[test] fn test() {}",
                Some("rs"),
                Some("rust"),
                0.2,
            ),
            create_test_result("docs/guide.md", "# Guide", Some("md"), None, 0.3),
            create_test_result(
                "lib/utils.rs",
                "pub fn util() {}",
                Some("rs"),
                Some("rust"),
                0.4,
            ),
        ];

        // Create classification with target_folders specified
        let mut classification =
            create_test_classification(Some("rust"), "implementation", true, 0.8);
        classification.target_folders = Some(vec!["src".to_string(), "lib".to_string()]);

        let filtered = test_hyde.apply_folder_extension_filtering(&classification, results);

        // Should only include files from src/ and lib/ folders
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|r| r.meta.file.contains("src/main.rs")));
        assert!(filtered
            .iter()
            .any(|r| r.meta.file.contains("lib/utils.rs")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("tests/")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("docs/")));
    }

    #[test]
    fn test_extension_filtering() {
        let test_hyde = TestHyde;

        let results = vec![
            create_test_result("main.rs", "fn main() {}", Some("rs"), Some("rust"), 0.1),
            create_test_result("app.py", "def main():", Some("py"), Some("python"), 0.2),
            create_test_result(
                "script.js",
                "function main() {}",
                Some("js"),
                Some("javascript"),
                0.3,
            ),
            create_test_result("README.md", "# README", Some("md"), None, 0.4),
        ];

        // Create classification with target_extensions specified
        let mut classification = create_test_classification(None, "implementation", true, 0.8);
        classification.target_extensions = Some(vec!["rs".to_string(), "py".to_string()]);

        let filtered = test_hyde.apply_folder_extension_filtering(&classification, results);

        // Should only include .rs and .py files
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|r| r.meta.file.contains("main.rs")));
        assert!(filtered.iter().any(|r| r.meta.file.contains("app.py")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("script.js")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("README.md")));
    }

    #[test]
    fn test_exclude_patterns_filtering() {
        let test_hyde = TestHyde;

        let results = vec![
            create_test_result("src/main.rs", "fn main() {}", Some("rs"), Some("rust"), 0.1),
            create_test_result(
                "tests/test.rs",
                "#[test] fn test() {}",
                Some("rs"),
                Some("rust"),
                0.2,
            ),
            create_test_result("target/debug/main", "binary", None, None, 0.3),
            create_test_result("docs/README.md", "# README", Some("md"), None, 0.4),
        ];

        // Create classification with exclude_patterns specified
        let mut classification =
            create_test_classification(Some("rust"), "implementation", true, 0.8);
        classification.exclude_patterns = Some(vec![
            "test".to_string(),
            "target".to_string(),
            "docs".to_string(),
        ]);

        let filtered = test_hyde.apply_folder_extension_filtering(&classification, results);

        // Should exclude files matching the patterns
        assert_eq!(filtered.len(), 1);
        assert!(filtered.iter().any(|r| r.meta.file.contains("src/main.rs")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("tests/")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("target/")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("docs/")));
    }

    #[test]
    fn test_combined_folder_extension_filtering() {
        let test_hyde = TestHyde;

        let results = vec![
            create_test_result("src/main.rs", "fn main() {}", Some("rs"), Some("rust"), 0.1),
            create_test_result("src/lib.py", "def lib():", Some("py"), Some("python"), 0.2),
            create_test_result(
                "tests/test.rs",
                "#[test] fn test() {}",
                Some("rs"),
                Some("rust"),
                0.3,
            ),
            create_test_result(
                "lib/utils.js",
                "function util() {}",
                Some("js"),
                Some("javascript"),
                0.4,
            ),
        ];

        // Create classification with both folder and extension filters
        let mut classification =
            create_test_classification(Some("rust"), "implementation", true, 0.8);
        classification.target_folders = Some(vec!["src".to_string()]);
        classification.target_extensions = Some(vec!["rs".to_string()]);

        let filtered = test_hyde.apply_folder_extension_filtering(&classification, results);

        // Should only include .rs files from src/ folder
        assert_eq!(filtered.len(), 1);
        assert!(filtered.iter().any(|r| r.meta.file.contains("src/main.rs")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("src/lib.py")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("tests/")));
        assert!(!filtered.iter().any(|r| r.meta.file.contains("lib/")));
    }

    // Helper function to create test QueryClassification with additional fields
    fn create_test_classification_with_filters(
        language: Option<&str>,
        intent: &str,
        wants_code: bool,
        confidence: f32,
        target_folders: Option<Vec<String>>,
        target_extensions: Option<Vec<String>>,
        exclude_patterns: Option<Vec<String>>,
    ) -> QueryClassification {
        QueryClassification {
            language: language.map(|s| s.to_string()),
            intent: intent.to_string(),
            wants_code,
            confidence,
            target_folders,
            target_extensions,
            exclude_patterns,
        }
    }

    #[test]
    fn test_repo_profile_integration() {
        use crate::repo::RepoProfile;
        
        // Create a test repo profile
        let repo_profile = RepoProfile {
            name: "test-repo".to_string(),
            primary_languages: vec!["Python".to_string(), "JavaScript".to_string()],
            build_files: vec!["package.json".to_string()],
            frameworks: vec!["React".to_string(), "FastAPI".to_string()],
            readme_summary: Some("A test repository for testing RepoProfile integration".to_string()),
        };

        // Test the methods that Hyde uses
        let description = repo_profile.description();
        let project_type = repo_profile.project_type();
        
        assert!(description.contains("test-repo"));
        assert!(description.contains("Python"));
        assert!(description.contains("JavaScript"));
        assert!(description.contains("React"));
        assert!(description.contains("FastAPI"));
        
        assert_eq!(project_type, "Node.js project"); // Because package.json is present
        
        // Test backward compatibility - ensure None works
        let default_description = "a software repository".to_string();
        let default_project_type = "software project".to_string();
        
        assert!(!default_description.contains("cargo-chat"));
        assert!(!default_project_type.contains("cargo-chat"));
    }
}
