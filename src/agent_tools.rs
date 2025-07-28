use anyhow::Result;
use serde_json::{json, Value};
use std::sync::Arc;
use crate::{
    ann::{ChunkMeta, DynamicAnn},
    embedding::Embedder,
    hyde::Hyde,
    llm::Tool,
    openai::OpenAIClient,
    repo::RepoProfile,
};
use futures_util::StreamExt;

pub struct AgentToolsState {
    pub embedder: Option<Arc<Embedder>>,
    pub ann_index: Option<DynamicAnn<ChunkMeta>>,
    pub index_path: Option<String>,
    pub repo_profile: Option<RepoProfile>,
    pub openai_client: Option<OpenAIClient>,
}

impl AgentToolsState {
    pub fn new() -> Self {
        Self {
            embedder: None,
            ann_index: None,
            index_path: None,
            repo_profile: None,
            openai_client: None,
        }
    }

    pub fn set_index(&mut self, embedder: Arc<Embedder>, ann_index: DynamicAnn<ChunkMeta>, index_path: String, repo_profile: Option<RepoProfile>) {
        self.embedder = Some(embedder);
        self.ann_index = Some(ann_index);
        self.index_path = Some(index_path);
        self.repo_profile = repo_profile;
    }

    pub fn set_openai_client(&mut self, client: OpenAIClient) {
        self.openai_client = Some(client);
    }
}

pub fn build_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "load_codebase_index",
            description: "Load an indexed codebase for querying. This must be called before using other codebase tools.",
            schema: json!({
                "type": "object",
                "properties": {
                    "index_path": {
                        "type": "string",
                        "description": "Path to the directory containing the indexed codebase (index.bin file)"
                    }
                },
                "required": ["index_path"]
            }),
        },
        Tool {
            name: "search_codebase",
            description: "Search through the loaded codebase using semantic similarity to find relevant code chunks",
            schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what code you're looking for"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of code chunks to return (default: 5, max: 20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "ask_codebase",
            description: "Ask a question about the loaded codebase and get a comprehensive answer with code references using HyDE (Hypothetical Document Embeddings)",
            schema: json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question about the codebase functionality, architecture, or implementation"
                    },
                    "num_references": {
                        "type": "integer",
                        "description": "Number of code references to include (default: 5, max: 10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["question"]
            }),
        },
        Tool {
            name: "get_codebase_info",
            description: "Get information about the currently loaded codebase including repository profile and statistics",
            schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        },
    ]
}

pub async fn exec_tool(tool_name: &str, input: &Value, state: &mut AgentToolsState) -> Result<String> {
    match tool_name {
        "load_codebase_index" => {
            let index_path = input["index_path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("index_path parameter required"))?;

            // Load embedder (using default Qwen3 model)
            let embedding_model = crate::embedding::EmbeddingModel::default_qwen3();
            let embedder = Arc::new(crate::embedding::Embedder::new(embedding_model)?);

            // Load ANN index
            let index_file_path = format!("{}/index.bin", index_path);
            let bytes = std::fs::read(&index_file_path)
                .map_err(|e| anyhow::anyhow!("Failed to read index file {}: {}", index_file_path, e))?;

            let ann_data = if let Ok(ann_512) = serde_json::from_slice::<crate::ann::Ann<512, ChunkMeta>>(&bytes) {
                DynamicAnn::Dim512(ann_512)
            } else if let Ok(ann_1024) = serde_json::from_slice::<crate::ann::Ann<1024, ChunkMeta>>(&bytes) {
                DynamicAnn::Dim1024(ann_1024)
            } else {
                return Err(anyhow::anyhow!("Failed to deserialize ANN index. Unsupported dimension."));
            };

            // Try to load repository profile
            let profile_path = format!("{}/profile.json", index_path);
            let repo_profile = if std::path::Path::new(&profile_path).exists() {
                std::fs::read_to_string(&profile_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<RepoProfile>(&s).ok())
            } else {
                None
            };

            // Initialize OpenAI client if API key is available
            if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
                let mut client = OpenAIClient::new(&api_key).with_model("gpt-4o-mini");
                if let Ok(api_url) = std::env::var("OPENAI_API_URL") {
                    client = client.with_api_url(&api_url);
                }
                state.set_openai_client(client);
            }

            state.set_index(embedder, ann_data, index_path.to_string(), repo_profile);

            if let Some(profile) = &state.repo_profile {
                Ok(format!("Successfully loaded codebase index for '{}' - {}. Frameworks: {}.", 
                    profile.name, 
                    profile.project_type(),
                    if profile.frameworks.is_empty() { "None".to_string() } else { profile.frameworks.join(", ") }))
            } else {
                Ok(format!("Successfully loaded codebase index from {}. Ready for queries.", index_path))
            }
        }

        "search_codebase" => {
            let query = input["query"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("query parameter required"))?;
            let num_results = input["num_results"].as_u64().unwrap_or(5) as usize;

            let embedder = state.embedder.as_ref()
                .ok_or_else(|| anyhow::anyhow!("No codebase loaded. Please call load_codebase_index first."))?;
            let ann_index = state.ann_index.as_ref()
                .ok_or_else(|| anyhow::anyhow!("No codebase index loaded."))?;

            // Generate embedding for the query
            let query_embedding = embedder.embed(query).await?;

            // Search the index
            let results = ann_index.query(&query_embedding, num_results as i32)?;

            let mut response = format!("Found {} relevant code chunks for query: \"{}\"\n\n", results.len(), query);
            
            for (i, result) in results.iter().enumerate() {
                let meta = &result.metadata;
                let snippet = if meta.code.len() > 300 {
                    format!("{}...", &meta.code[..300])
                } else {
                    meta.code.clone()
                };

                response.push_str(&format!(
                    "{}. File: {} ({})\n   Distance: {:.4}\n   Code:\n{}\n\n",
                    i + 1,
                    meta.file,
                    meta.language.as_deref().unwrap_or("unknown"),
                    result.distance,
                    snippet
                ));
            }

            Ok(response)
        }

        "ask_codebase" => {
            let question = input["question"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("question parameter required"))?;
            let num_references = input["num_references"].as_u64().unwrap_or(5) as usize;

            let embedder = state.embedder.as_ref()
                .ok_or_else(|| anyhow::anyhow!("No codebase loaded. Please call load_codebase_index first."))?;
            let ann_index = state.ann_index.as_ref()
                .ok_or_else(|| anyhow::anyhow!("No codebase index loaded."))?;
            let openai_client = state.openai_client.as_ref()
                .ok_or_else(|| anyhow::anyhow!("OpenAI client not configured. Set OPENAI_API_KEY environment variable."))?;

            // Create Hyde instance
            let answer_client = openai_client.clone().with_model("gpt-4o");
            let hyde_client = openai_client.clone().with_model("gpt-4o-mini");
            
            let hyde = Hyde::new(
                hyde_client,
                answer_client,
                embedder.clone(),
                ann_index,
                1000,
                None, // No reranker for now
                state.repo_profile.clone(),
            );

            // Get comprehensive answer with Hyde
            let mut hyde_response = hyde.retrieve(question, num_references, false).await?;

            // Collect the streaming answer
            let mut full_answer = String::new();
            while let Some(chunk_result) = hyde_response.answer_stream.next().await {
                match chunk_result {
                    Ok(chunk) => full_answer.push_str(&chunk),
                    Err(e) => return Err(anyhow::anyhow!("Error in answer stream: {}", e)),
                }
            }

            // Format response with references
            let mut response = format!("## Answer\n{}\n\n", full_answer);

            if !hyde_response.code_refs.is_empty() {
                response.push_str("## Code References\n");
                for (i, result) in hyde_response.code_refs.iter().enumerate() {
                    let meta = &result.meta;
                    let snippet = if meta.code.len() > 400 {
                        format!("{}...", &meta.code[..400])
                    } else {
                        meta.code.clone()
                    };

                    response.push_str(&format!(
                        "{}. **{}** ({})\n   Distance: {:.4}\n   ```{}\n{}\n   ```\n\n",
                        i + 1,
                        meta.file,
                        meta.language.as_deref().unwrap_or("unknown"),
                        result.distance,
                        meta.language.as_deref().unwrap_or(""),
                        snippet
                    ));
                }
            }

            Ok(response)
        }

        "get_codebase_info" => {
            if state.ann_index.is_none() {
                return Ok("No codebase currently loaded. Use load_codebase_index to load a codebase.".to_string());
            }

            let mut response = format!("## Codebase Information\n");
            
            if let Some(path) = &state.index_path {
                response.push_str(&format!("**Index Path:** {}\n", path));
            }

            if let Some(profile) = &state.repo_profile {
                response.push_str(&format!(
                    "**Repository:** {}\n**Project Type:** {}\n**Primary Languages:** {}\n**Frameworks:** {}\n\n",
                    profile.name,
                    profile.project_type(),
                    if profile.primary_languages.is_empty() { "None detected".to_string() } else { profile.primary_languages.join(", ") },
                    if profile.frameworks.is_empty() { "None detected".to_string() } else { profile.frameworks.join(", ") }
                ));

                if !profile.build_files.is_empty() {
                    response.push_str("**Build Files:**\n");
                    for build_file in &profile.build_files {
                        response.push_str(&format!("- {}\n", build_file));
                    }
                }
            }

            if let Some(ann_index) = &state.ann_index {
                let chunk_count = match ann_index {
                    DynamicAnn::Dim512(ann) => ann.metadata.len(),
                    DynamicAnn::Dim1024(ann) => ann.metadata.len(),
                };
                response.push_str(&format!("\n**Indexed Chunks:** {}\n", chunk_count));
            }

            Ok(response)
        }

        _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
    }
}
