use clap::{Parser, Subcommand, ValueEnum};
use anyhow::Result;
use log::{info, error, warn};
use serde_json;
use tracing_subscriber::{fmt, fmt::format::FmtSpan, prelude::*, EnvFilter};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressStyle};
use futures::future::join_all;
use futures_util::StreamExt;
use std::io::{stdout, Write};
use std::str::FromStr;

mod chunker; mod embedding; mod ann; mod rerank; mod hyde;
mod openai; mod language; mod repo; mod llm; mod agent_tools;
use ann::ChunkMeta;
use embedding::EmbeddingModel;
use repo::RepoProfile;

#[derive(Debug, Clone, ValueEnum)]
pub enum EmbeddingModelType {
    Jina,
    Qwen3,
}

impl EmbeddingModelType {
    pub fn to_embedding_model(&self) -> EmbeddingModel {
        match self {
            EmbeddingModelType::Jina => EmbeddingModel::default_jina(),
            EmbeddingModelType::Qwen3 => EmbeddingModel::default_qwen3(),
        }
    }
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Index a code repository: chunk files, generate embeddings, and build an ANN index.
    Index {
        repo: String,
        out: String,
        /// Custom model ID (overrides --model-type)
        #[arg(long)]
        model_id: Option<String>,
        /// Predefined embedding model type
        #[arg(long, value_enum, default_value_t = EmbeddingModelType::Qwen3)]
        model_type: EmbeddingModelType,
    },
    /// Query a previously built index using a question to find relevant code and synthesize an answer.
    Query {
        index_dir: String,
        /// Custom model ID (overrides --model-type)
        #[arg(long)]
        model_id: Option<String>,
        /// Predefined embedding model type
        #[arg(long, value_enum, default_value_t = EmbeddingModelType::Qwen3)]
        model_type: EmbeddingModelType,
        q: String,
        k: usize,
        #[arg(long)]
        rerank_model: Option<String>,
        #[arg(long, action = clap::ArgAction::SetTrue)]
        use_rerank: bool,
        /// Model for Hyde queries (default: gpt-4o-mini)
        #[arg(long, default_value = "gpt-4o-mini")]
        hyde_model: String,
        /// Model for final answer synthesis (default: gpt-4o)
        #[arg(long, default_value = "gpt-4o")]
        answer_model: String,
    },
    /// Start an interactive REPL session for efficient iterative indexing and querying.
    Interactive { 
        /// Custom model ID (overrides --model-type)
        #[arg(long)]
        model_id: Option<String>,
        /// Predefined embedding model type
        #[arg(long, value_enum, default_value_t = EmbeddingModelType::Qwen3)]
        model_type: EmbeddingModelType,
        /// Model for Hyde queries (default: gpt-4o-mini)
        #[arg(long, default_value = "gpt-4o-mini")]
        hyde_model: String,
        /// Model for final answer synthesis (default: gpt-4o)
        #[arg(long, default_value = "gpt-4o")]
        answer_model: String,
    },
    /// Start an AI agent that can answer questions about indexed codebases using tools.
    Agent {
        /// LLM backend to use (anthropic or openai)
        #[arg(long, default_value = "anthropic")]
        backend: String,
        /// Custom model name (optional)
        #[arg(long)]
        model: Option<String>,
    },
}

#[derive(Parser, Debug)]
#[clap(no_binary_name = true, version = "1.0", author = "REPL", disable_help_subcommand = true)]
struct ReplCmdParser {
    #[command(subcommand)]
    cmd: ReplSubCmd,
}

#[derive(Subcommand, Debug)]
enum ReplSubCmd {
    /// Index a repository using the loaded embedding model
    Index(ReplIndexArgs),
    /// Load an existing ANN index into the session
    LoadIndex(ReplLoadIndexArgs),
    /// Query the loaded index using the loaded embedding model
    Query(ReplQueryArgs),
    /// Show current session status (e.g., loaded model, index)
    Status,
    /// Show help for REPL commands
    Help,
    /// Exit the interactive session
    Exit,
}

#[derive(Parser, Debug)]
struct ReplIndexArgs {
    #[arg(long)]
    repo: String,
    #[arg(long)]
    out: String,
    /// Custom model ID (overrides --model-type)
    #[arg(long)]
    model_id: Option<String>,
    /// Predefined embedding model type
    #[arg(long, value_enum, default_value_t = EmbeddingModelType::Qwen3)]
    model_type: EmbeddingModelType,
}

#[derive(Parser, Debug)]
struct ReplLoadIndexArgs {
    index_dir: String,
}

#[derive(Parser, Debug)]
struct ReplQueryArgs {
    query_parts: Vec<String>,
    #[arg(short, long, default_value_t = 3)]
    k: usize,
    #[arg(long)]
    rerank_model: Option<String>,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    use_rerank: bool,
    /// Model for Hyde queries (default: gpt-4o-mini)
    #[arg(long, default_value = "gpt-4o-mini")]
    hyde_model: String,
    /// Model for final answer synthesis (default: gpt-4o)
    #[arg(long, default_value = "gpt-4o")]
    answer_model: String,
}

struct SessionState {
    embedder: Arc<embedding::Embedder>,
    ann_index: Option<ann::DynamicAnn<ChunkMeta>>,
    model: EmbeddingModel,
    current_index_path: Option<String>,
}

impl SessionState {
    fn new(model: EmbeddingModel, embedder: Arc<embedding::Embedder>) -> Self {
        Self {
            embedder,
            ann_index: None,
            model,
            current_index_path: None,
        }
    }
}

/// Resolve embedding model from CLI arguments
fn resolve_embedding_model(model_id: Option<String>, model_type: EmbeddingModelType) -> Result<EmbeddingModel> {
    if let Some(id) = model_id {
        // Custom model ID takes precedence
        EmbeddingModel::from_str(&id)
    } else {
        // Use predefined model type
        Ok(model_type.to_embedding_model())
    }
}

/// Create a RepoProfile by looking for the source directory that was indexed
async fn create_repo_profile_from_index_dir(index_dir: &str) -> Option<RepoProfile> {
    let index_path = match std::env::current_dir() {
        Ok(current) => current.join(index_dir),
        Err(_) => std::path::Path::new(index_dir).to_path_buf(),
    };
    
    // First try to load cached profile.json
    let cached_path = index_path.join("profile.json");
    if cached_path.exists() {
        match std::fs::read_to_string(&cached_path)
            .ok()
            .and_then(|s| serde_json::from_str::<RepoProfile>(&s).ok())
        {
            Some(profile) => {
                info!("Loaded cached RepoProfile from {}", cached_path.display());
                return Some(profile);
            }
            None => {
                warn!("profile.json exists but could not be parsed - regenerating...");
            }
        }
    }
    
    // Fallback to old (expensive) logic - try to find the original source directory
    // First try the parent directory (common case: repo/index_output)
    if let Some(parent) = index_path.parent() {
        if has_source_files(parent) {
            match RepoProfile::from_directory(parent).await {
                Ok(profile) => {
                    info!("Created repository profile for: {}", profile.name);
                    return Some(profile);
                }
                Err(e) => {
                    warn!("Failed to create RepoProfile from parent directory {}: {}", parent.display(), e);
                }
            }
        }
    }
    
    // Then try the index directory itself (if it contains source files)
    if has_source_files(&index_path) {
        match RepoProfile::from_directory(&index_path).await {
            Ok(profile) => {
                info!("Created repository profile for: {}", profile.name);
                return Some(profile);
            }
            Err(e) => {
                warn!("Failed to create RepoProfile from index directory {}: {}", index_path.display(), e);
            }
        }
    }
    
    warn!("Could not create RepoProfile - no source files found near index directory: {}", index_dir);
    None
}

/// Check if a directory contains source files (indicating it's likely a repository root)
fn has_source_files(path: &std::path::Path) -> bool {
    if !path.exists() || !path.is_dir() {
        return false;
    }
    
    // Check for common indicators of a source repository
    let indicators = [
        "src", "lib", "main.rs", "lib.rs", "Cargo.toml", "package.json", 
        "pom.xml", "go.mod", "pyproject.toml", "requirements.txt"
    ];
    
    for indicator in &indicators {
        let indicator_path = path.join(indicator);
        if indicator_path.exists() {
            return true;
        }
    }
    
    // Check for common source file extensions in the root
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(extension) = entry.path().extension() {
                if let Some(ext_str) = extension.to_str() {
                    if matches!(ext_str, "rs" | "py" | "js" | "ts" | "java" | "go" | "cpp" | "c" | "h") {
                        return true;
                    }
                }
            }
        }
    }
    
    false
}

async fn execute_index_command(
    embedder: &embedding::Embedder,
    repo_path: &str,
    output_dir: &str,
) -> Result<()> {
    let overall_start_time = std::time::Instant::now(); // Record start time for the entire operation

    info!("Starting chunking for repo: {}", repo_path);
    let chunks = chunker::chunk_repo(repo_path)?;
    
    let mut vecs = Vec::new();
    let mut metas = Vec::new();
    
    info!("Embedding {} chunks...", chunks.len());
    let app_batch_size = 32; 

    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-~");

    let pb = Arc::new(ProgressBar::new(chunks.len() as u64));
    pb.set_style(pb_style);

    let mut processing_futures = Vec::new();

    for (batch_idx, chunk_batch_slice) in chunks.chunks(app_batch_size).enumerate() {
        let owned_chunk_batch: Vec<chunker::CodeChunk> = chunk_batch_slice.to_vec();
        let pb_clone = Arc::clone(&pb);
        
        // The embedder reference can be used directly as it's Sync
        // and the tasks are awaited within this function's scope.
        let future = async move {
            let texts_to_embed: Vec<String> = owned_chunk_batch.iter().map(|chunk| chunk.content.clone()).collect();
            let batch_len = owned_chunk_batch.len() as u64; // Get length before potential move
            
            let result = if texts_to_embed.is_empty() {
                Ok((owned_chunk_batch, Vec::new()))
            } else {
                let text_slices: Vec<&str> = texts_to_embed.iter().map(AsRef::as_ref).collect();
                match embedder.embed_batch(&text_slices, Some(text_slices.len())).await {
                    Ok(embeddings) => Ok((owned_chunk_batch, embeddings)),
                    Err(e) => Err(anyhow::anyhow!("Error in batch {}: {}", batch_idx, e)),
                }
            };
            
            // Increment progress regardless of outcome for this batch's items
            pb_clone.inc(batch_len); // Use the stored length
            result
        };
        processing_futures.push(future);
    }

    let all_results = join_all(processing_futures).await;
    
    // Finish progress bar after all tasks are processed (successfully or not)
    // but before potentially erroring out from results processing.
    pb.finish_with_message("Embedding processing attempted for all chunks.");

    for result_item in all_results {
        match result_item {
            Ok((original_chunk_batch, embedding_arrays)) => {
                if !embedding_arrays.is_empty() {
                    for (embedding_idx, embedding_vector) in embedding_arrays.iter().enumerate() {
                        let chunk = &original_chunk_batch[embedding_idx];
                        vecs.push(embedding_vector.clone());
                        metas.push(ChunkMeta { 
                            file: chunk.file_path.clone(), 
                            code: chunk.content.clone(),
                            language: chunk.language.clone(),
                            extension: chunk.extension.clone(),
                        });
                    }
                }
            }
            Err(e) => {
                error!("Failed to process a batch of embeddings: {}", e);
                // It's important to decide error strategy: continue with partial embeddings or fail hard.
                // For now, propagating the first error encountered.
                return Err(e.into()); 
            }
        }
    }
    
    tracing::info!("Embeddings generation phase complete. {} embeddings collected.", vecs.len());

    if vecs.is_empty() {
        warn!("No embeddings were generated. Index will be empty.");
    }
    
    let ann_instance = ann::DynamicAnn::<ChunkMeta>::build(vecs, metas)?;
    
    std::fs::create_dir_all(output_dir)?;
    let index_file_path = format!("{}/index.bin", output_dir);
    info!("Serializing and writing ANN index to {}", index_file_path);
    std::fs::write(&index_file_path, serde_json::to_vec(&ann_instance)?)?;
    info!("ANN index successfully built and saved to {}", output_dir);

    // Generate and save repository profile
    info!("Generating repository profile...");
    let profile = repo::RepoProfile::from_directory(std::path::Path::new(repo_path)).await?;
    let profile_path = format!("{}/profile.json", output_dir);
    std::fs::write(&profile_path, serde_json::to_string_pretty(&profile)?)?;
    info!("Repository profile saved to {}", profile_path);

    let overall_duration = overall_start_time.elapsed();
    println!("Total indexing time: {:.2?}", overall_duration); // Display total time

    Ok(())
}

async fn print_query_results(mut hits: hyde::HydeResponse) -> Result<()> {
    let mut full_answer_empty = true;
    while let Some(chunk_result) = hits.answer_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                if !chunk.is_empty() {
                    full_answer_empty = false;
                }
                print!("{}", chunk);
                if let Err(e) = stdout().flush() {
                    warn!("Failed to flush stdout: {}", e);
                }
            }
            Err(e) => {
                error!("Error receiving chunk from answer stream: {}", e);
                eprintln!("\n[Error in answer stream: {}]", e);
                break;
            }
        }
    }
    
    if !full_answer_empty {
        println!(); 
    }

    if !hits.code_refs.is_empty() {
        println!("\nReferences:");
        for (rank, res) in hits.code_refs.iter().enumerate() {
            let file = &res.meta.file;
            let code = &res.meta.code;
            let snippet = code.trim();
            let snippet_display = if snippet.len() > 200 {
                format!("{}...", &snippet[..200])
            } else {
                snippet.to_string()
            };
            println!(
                "{}. File: {}
   Chunk ID: {}
   Distance: {:.4}
   Code:
{}
",
                rank + 1,
                file,
                res.index,
                res.distance,
                snippet_display
            );
        }
    } else {
        println!("No specific code references found for this answer.");
    }
    Ok(())
}

async fn execute_query_command(
    embedder: Arc<embedding::Embedder>,
    ann_index: &ann::DynamicAnn<ChunkMeta>,
    index_dir: &str,
    rerank_model_path: Option<&str>,
    query_string: &str,
    k: usize,
    use_rerank_flag: bool,
    hyde_model: &str,
    answer_model: &str,
) -> Result<()> {
    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set. This key is required for the query command."))?;
    
    let openai_api_url = std::env::var("OPENAI_API_URL").ok();
    
    // Create client for Hyde queries (cheaper model)
    let hyde_client = {
        let client = openai::OpenAIClient::new(&openai_api_key)
            .with_model(hyde_model);
        if let Some(url) = openai_api_url.as_ref() {
            client.with_api_url(url)
        } else {
            client
        }
    };
    
    // Create client for final answers (better model)
    let answer_client = {
        let client = openai::OpenAIClient::new(&openai_api_key)
            .with_model(answer_model);
        if let Some(url) = openai_api_url.as_ref() {
            client.with_api_url(url)
        } else {
            client
        }
    };

    let reranker_instance = if use_rerank_flag {
        match rerank_model_path {
            Some(path) => Some(rerank::Reranker::new(path, ())?),
            None => {
                warn!("Reranking is enabled, but no rerank model path was provided. Proceeding without reranking.");
                None
            }
        }
    } else {
        None
    };
    
    let answer_model_name = answer_client.model.clone(); // Get model name for progress display

    let effective_use_rerank = use_rerank_flag && reranker_instance.is_some();

    // Create repository profile for better context
    let repo_profile = create_repo_profile_from_index_dir(index_dir).await;

    let hyde = hyde::Hyde::new(hyde_client, answer_client, embedder, ann_index, 1000, reranker_instance, repo_profile);
    
    // Start progress bar
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&[".", "..", "...", "....", "....."])
            .template("Generating answer with {msg} {spinner:.blue}")
            .unwrap(),
    );
    pb.set_message(answer_model_name.clone());

    let start_time = std::time::Instant::now();

    info!("Retrieving results for query: '{}' with k={}", query_string, k);
    // hits_result is now Result<HydeResponse> where HydeResponse contains the stream
    let hits_result = hyde.retrieve(query_string, k, effective_use_rerank).await;
    
    let duration = start_time.elapsed();
    pb.finish_and_clear(); // Clear the progress bar

    match hits_result {
        Ok(hits) => { // hits is HydeResponse, containing the stream
            print_query_results(hits).await?; 
            println!("Answer generated by {} in {:.2?}:", answer_model_name, duration);
            Ok(())
        }
        Err(e) => {
            error!("Error retrieving results: {}", e);
            Err(e)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from a local .env file if present (non-fatal when absent)
    let _ = dotenvy::dotenv();
    tracing_subscriber::registry()
        .with(fmt::layer().with_span_events(FmtSpan::CLOSE))
        .with(EnvFilter::from_default_env())
        .init();

    let args = Cli::parse();
    match args.cmd {
        Cmd::Index { repo, out, model_id, model_type } => {
            let embedding_model = resolve_embedding_model(model_id, model_type)?;
            info!("Loading embedder with model: {}", embedding_model);
            let embedder = embedding::Embedder::new(embedding_model)?;
            execute_index_command(&embedder, &repo, &out).await?;
        }
        Cmd::Query { index_dir, model_id, model_type, q, k, rerank_model, use_rerank, hyde_model, answer_model } => {
            let embedding_model = resolve_embedding_model(model_id, model_type)?;
            info!("Loading embedder with model: {}", embedding_model);
            let embedder = Arc::new(embedding::Embedder::new(embedding_model)?); 
            
            let index_file_path = format!("{}/index.bin", index_dir);
            info!("Loading ANN index from: {}", index_file_path);
            let bytes = std::fs::read(&index_file_path)?;
            
            // Try to load as 512D first, then 1024D
            let ann_data = if let Ok(ann_512) = serde_json::from_slice::<ann::Ann<512, ChunkMeta>>(&bytes) {
                ann::DynamicAnn::Dim512(ann_512)
            } else if let Ok(ann_1024) = serde_json::from_slice::<ann::Ann<1024, ChunkMeta>>(&bytes) {
                ann::DynamicAnn::Dim1024(ann_1024)
            } else {
                return Err(anyhow::anyhow!("Failed to deserialize ANN index. Unsupported dimension."));
            };
            
            execute_query_command(embedder, &ann_data, &index_dir, rerank_model.as_deref(), &q, k, use_rerank, &hyde_model, &answer_model).await?;
        }
        Cmd::Interactive { model_id, model_type, hyde_model, answer_model } => {
            let embedding_model = resolve_embedding_model(model_id, model_type)?;
            info!("Starting interactive session. Loading embedder with model: {}", embedding_model);
            let embedder_instance = embedding::Embedder::new(embedding_model.clone())?;
            let shared_embedder = Arc::new(embedder_instance);
            let mut session_state = SessionState::new(embedding_model, shared_embedder.clone());
            
            let mut rl = DefaultEditor::new()?;

            #[cfg(feature = "with-file-history")]
            let history_file_path = {
                const HISTORY_FILE_NAME: &str = ".cargo_chat_history";
                let home_dir_opt = if cfg!(unix) {
                    std::env::var("HOME").ok()
                } else if cfg!(windows) {
                    std::env::var("USERPROFILE").ok()
                } else {
                    None
                };

                if let Some(home_dir) = home_dir_opt {
                    let mut path_buf = std::path::PathBuf::from(home_dir);
                    path_buf.push(HISTORY_FILE_NAME);
                    path_buf
                } else {
                    warn!("Could not determine home directory. Using current directory for history: {}", HISTORY_FILE_NAME);
                    std::path::PathBuf::from(HISTORY_FILE_NAME)
                }
            };

            #[cfg(feature = "with-file-history")]
            if rl.load_history(&history_file_path).is_err() {
                info!("No previous history found or error loading from {:?}.", history_file_path);
            }

            println!("Interactive Cargo Chat session (Model: {}). Type 'help' for commands, 'exit' to quit.", session_state.model);

            loop {
                let prompt_text = format!("cargo-chat ({})> ", session_state.current_index_path.as_deref().unwrap_or("no index"));
                let readline = rl.readline(&prompt_text);
                match readline {
                    Ok(line) => {
                        rl.add_history_entry(line.as_str())?;
                        let parts: Vec<&str> = line.trim().split_whitespace().collect();
                        if parts.is_empty() {
                            continue;
                        }

                        match ReplCmdParser::try_parse_from(parts) {
                            Ok(repl_cli) => {
                                match repl_cli.cmd {
                                    ReplSubCmd::Index(args) => {
                                        info!("Executing REPL Index: repo={}, out={}", args.repo, args.out);
                                        let embedding_model = resolve_embedding_model(args.model_id.clone(), args.model_type.clone())?;
                                        let embedder = embedding::Embedder::new(embedding_model)?;
                                        match execute_index_command(&embedder, &args.repo, &args.out).await {
                                            Ok(()) => {
                                                info!("Index command completed successfully. Attempting to load the new index.");
                                                let index_file_path = format!("{}/index.bin", args.out);
                                                match std::fs::read(&index_file_path) {
                                                    Ok(bytes) => {
                                                        // Try to load as 512D first, then 1024D
                                                        let ann_data = if let Ok(ann_512) = serde_json::from_slice::<ann::Ann<512, ChunkMeta>>(&bytes) {
                                                            ann::DynamicAnn::Dim512(ann_512)
                                                        } else if let Ok(ann_1024) = serde_json::from_slice::<ann::Ann<1024, ChunkMeta>>(&bytes) {
                                                            ann::DynamicAnn::Dim1024(ann_1024)
                                                        } else {
                                                            error!("Failed to deserialize newly created index. Unsupported dimension.");
                                                            continue;
                                                        };
                                                        session_state.ann_index = Some(ann_data);
                                                        session_state.current_index_path = Some(args.out.clone());
                                                        info!("Successfully loaded newly created index from {}", args.out);
                                                    },
                                                    Err(e) => error!("Failed to read newly created index file {}: {}", index_file_path, e),
                                                }
                                            }
                                            Err(e) => error!("Error executing index command: {}", e),
                                        }
                                    }
                                    ReplSubCmd::LoadIndex(args) => {
                                        let index_file_path = format!("{}/index.bin", args.index_dir);
                                        info!("REPL: Loading ANN index from: {}", index_file_path);
                                        match std::fs::read(&index_file_path) {
                                            Ok(bytes) => {
                                                // Try to load as 512D first, then 1024D
                                                let ann_data = if let Ok(ann_512) = serde_json::from_slice::<ann::Ann<512, ChunkMeta>>(&bytes) {
                                                    ann::DynamicAnn::Dim512(ann_512)
                                                } else if let Ok(ann_1024) = serde_json::from_slice::<ann::Ann<1024, ChunkMeta>>(&bytes) {
                                                    ann::DynamicAnn::Dim1024(ann_1024)
                                                } else {
                                                    error!("Failed to deserialize index. Unsupported dimension.");
                                                    continue;
                                                };
                                                session_state.ann_index = Some(ann_data);
                                                session_state.current_index_path = Some(args.index_dir.clone());
                                                info!("Index loaded successfully from {}", args.index_dir);
                                            },
                                            Err(e) => error!("Failed to read index file {}: {}", index_file_path, e),
                                        }
                                    }
                                    ReplSubCmd::Query(args) => {
                                        let final_query_string = args.query_parts.join(" ");
                                        info!("Executing REPL Query: '{}', k={}", final_query_string, args.k);
                                        if let Some(ref ann_data) = session_state.ann_index {
                                            if let Some(ref index_path) = session_state.current_index_path {
                                                if let Err(e) = execute_query_command(
                                                    session_state.embedder.clone(),
                                                    ann_data,
                                                    index_path,
                                                    args.rerank_model.as_deref(),
                                                    &final_query_string,
                                                    args.k,
                                                    args.use_rerank,
                                                    &args.hyde_model,
                                                    &args.answer_model,
                                                ).await {
                                                    error!("Error executing query command: {}", e);
                                                }
                                            } else {
                                                error!("No index path stored. This is a bug.");
                                            }
                                        } else {
                                            error!("No index loaded. Please use 'load_index <path>' first.");
                                        }
                                    }
                                    ReplSubCmd::Status => {
                                        println!("Session Status:");
                                        println!("  Model: {}", session_state.model);
                                        println!("  Embedder Loaded: Yes");
                                        if let Some(ref p) = session_state.current_index_path {
                                            println!("  Current Index: {} (Loaded)", p);
                                        } else {
                                            println!("  Current Index: Not loaded");
                                        }
                                    }
                                    ReplSubCmd::Help => {
                                        println!("Available REPL commands:");
                                        println!("  index --repo <path_to_repo> --out <output_dir_path>               : Indexes a repository.");
                                        println!("  load-index <index_dir_path>                                   : Loads an ANN index from the specified directory.");
                                        println!("  query \"<your question>\" [-k <num>] [--use-rerank] [--rerank-model <path>] [--synthesis-model <model>] : Queries the loaded index.");
                                        println!("  status                                                        : Shows current session status.");
                                        println!("  help                                                          : Shows this help message.");
                                        println!("  exit                                                          : Exits the interactive session.");
                                    }
                                    ReplSubCmd::Exit => {
                                        info!("Exiting interactive session.");
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Invalid REPL command or arguments: {}. Type 'help' for usage.", e.kind());
                                if e.kind() == clap::error::ErrorKind::DisplayHelp || e.kind() == clap::error::ErrorKind::DisplayVersion {
                                   println!("{}", e);
                                }
                            }
                        }
                    }
                    Err(ReadlineError::Interrupted) => {
                        info!("CTRL-C: To exit, type 'exit'.");
                    }
                    Err(ReadlineError::Eof) => {
                        info!("CTRL-D: Exiting interactive session.");
                        break;
                    }
                    Err(err) => {
                        error!("REPL Error: {:?}", err);
                        break;
                    }
                }
            }
            #[cfg(feature = "with-file-history")]
            if rl.save_history(&history_file_path).is_err() {
                error!("Failed to save REPL history to {:?}.", history_file_path);
            }
        }
        Cmd::Agent { backend, model } => {
            execute_agent_command(backend, model).await?;
        }
    }
    Ok(())
}

async fn execute_agent_command(backend: String, model: Option<String>) -> Result<()> {
    use llm::{Backend, Anthropic, OpenAi, Block, Msg};
    use agent_tools::{AgentToolsState, build_tools, exec_tool};
    use std::io::{self, Write};

    // Load environment variables
    dotenvy::dotenv().ok();

    // Create backend
    let llm_backend: Box<dyn Backend> = match backend.to_lowercase().as_str() {
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;
            let mut client = OpenAi::new(api_key);
            if let Some(m) = model {
                client = client.with_model(&m);
            }
            if let Ok(url) = std::env::var("OPENAI_API_URL") {
                client = client.with_api_url(&url);
            }
            Box::new(client)
        }
        _ => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;
            let mut client = Anthropic::new(api_key);
            if let Some(m) = model {
                client = client.with_model(&m);
            }
            Box::new(client)
        }
    };

    let mut chat: Vec<Msg> = Vec::new();
    let tools = build_tools();
    let mut agent_state = AgentToolsState::new();

    println!("🤖 Cargo Chat Agent - Codebase Q&A Assistant");
    println!("Backend: {}", backend);
    println!("Commands: Ask questions about codebases, or type 'quit' to exit");
    println!("First, load a codebase with: load the index from /path/to/index\n");

    loop {
        // Read user input
        print!("You: ");
        io::stdout().flush()?;
        let mut buf = String::new();
        if io::stdin().read_line(&mut buf)? == 0 { 
            break; 
        }
        let user_input = buf.trim();
        
        if user_input.is_empty() {
            continue;
        }
        
        if user_input == "quit" || user_input == "exit" {
            break;
        }

        // Add user message
        chat.push(Msg { 
            role: "user", 
            blocks: vec![Block::Text(user_input.to_string())] 
        });

        // Call LLM with streaming
        print!("🤖: ");
        io::stdout().flush()?;

        // Try streaming first, with robust fallback
        let streaming_result = llm_backend.chat_stream(&chat, &tools).await;
        
        match streaming_result {
            Ok(mut stream) => {
                let mut response_blocks = Vec::new();
                let mut has_text_output = false;
                let mut received_any_blocks = false;

                while let Some(block_result) = stream.next().await {
                    received_any_blocks = true;
                    match block_result {
                        Ok(block) => {
                            match block {
                                Block::Text(text) => {
                                    print!("{}", text);
                                    io::stdout().flush()?;
                                    has_text_output = true;
                                    response_blocks.push(Block::Text(text));
                                }
                                Block::ToolUse { id, name, input } => {
                                    // Execute tool
                                    match exec_tool(&name, &input, &mut agent_state).await {
                                        Ok(result) => {
                                            // Always display tool results
                                            print!("{}", result);
                                            io::stdout().flush()?;
                                            has_text_output = true;
                                            response_blocks.push(Block::ToolUse { id: id.clone(), name: name.clone(), input });
                                            response_blocks.push(Block::ToolResult { 
                                                id: id.clone(), 
                                                name: "tool_result".to_string(), 
                                                result: serde_json::Value::String(result) 
                                            });
                                        }
                                        Err(e) => {
                                            let error_msg = format!("Tool error: {}", e);
                                            print!("{}", error_msg);
                                            io::stdout().flush()?;
                                            response_blocks.push(Block::ToolUse { id: id.clone(), name: name.clone(), input });
                                            response_blocks.push(Block::ToolResult { 
                                                id: id.clone(), 
                                                name: "tool_result".to_string(), 
                                                result: serde_json::Value::String(error_msg) 
                                            });
                                        }
                                    }
                                }
                                Block::ToolResult { .. } => {
                                    // This shouldn't happen in assistant responses
                                    response_blocks.push(block);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Stream error: {}", e);
                            println!("\n[Error in stream: {}]", e);
                        }
                    }
                }

                if !received_any_blocks {
                    // If no blocks received, fall back to non-streaming
                    match llm_backend.chat(&chat, &tools).await {
                        Ok(blocks) => {
                            for block in blocks {
                                match block {
                                    Block::Text(text) => {
                                        print!("{}", text);
                                        io::stdout().flush()?;
                                        has_text_output = true;
                                        response_blocks.push(Block::Text(text));
                                    }
                                    Block::ToolUse { id, name, input } => {
                                        // Execute tool
                                        match exec_tool(&name, &input, &mut agent_state).await {
                                            Ok(result) => {
                                                // Always display tool results
                                                print!("{}", result);
                                                io::stdout().flush()?;
                                                has_text_output = true;
                                                response_blocks.push(Block::ToolUse { id: id.clone(), name: name.clone(), input });
                                                response_blocks.push(Block::ToolResult { 
                                                    id: id.clone(), 
                                                    name: "tool_result".to_string(), 
                                                    result: serde_json::Value::String(result) 
                                                });
                                            }
                                            Err(e) => {
                                                let error_msg = format!("Tool error: {}", e);
                                                print!("{}", error_msg);
                                                io::stdout().flush()?;
                                                response_blocks.push(Block::ToolUse { id: id.clone(), name: name.clone(), input });
                                                response_blocks.push(Block::ToolResult { 
                                                    id: id.clone(), 
                                                    name: "tool_result".to_string(), 
                                                    result: serde_json::Value::String(error_msg) 
                                                });
                                            }
                                        }
                                    }
                                    Block::ToolResult { .. } => {
                                        // This shouldn't happen in assistant responses
                                        response_blocks.push(block);
                                    }
                                }
                            }
                        }
                        Err(_e) => {
                            // Fallback also failed, but continue
                        }
                    }
                }
                println!(); // New line after response

                // Add assistant response to chat
                if !response_blocks.is_empty() {
                    chat.push(Msg { 
                        role: "assistant", 
                        blocks: response_blocks 
                    });
                }
            }
            Err(_e) => {
                // Fallback to non-streaming
                match llm_backend.chat(&chat, &tools).await {
                    Ok(blocks) => {
                        let mut response_blocks = Vec::new();
                        let mut has_text_output = false;

                        for block in blocks {
                            match block {
                                Block::Text(text) => {
                                    print!("{}", text);
                                    io::stdout().flush()?;
                                    has_text_output = true;
                                    response_blocks.push(Block::Text(text));
                                }
                                Block::ToolUse { id, name, input } => {
                                    // Execute tool
                                    match exec_tool(&name, &input, &mut agent_state).await {
                                        Ok(result) => {
                                            // Always display tool results in fallback too
                                            print!("{}", result);
                                            io::stdout().flush()?;
                                            has_text_output = true;
                                            response_blocks.push(Block::ToolUse { id: id.clone(), name: name.clone(), input });
                                            response_blocks.push(Block::ToolResult { 
                                                id: id.clone(), 
                                                name: "tool_result".to_string(), 
                                                result: serde_json::Value::String(result) 
                                            });
                                        }
                                        Err(e) => {
                                            let error_msg = format!("Tool error: {}", e);
                                            print!("{}", error_msg);
                                            io::stdout().flush()?;
                                            response_blocks.push(Block::ToolUse { id: id.clone(), name: name.clone(), input });
                                            response_blocks.push(Block::ToolResult { 
                                                id: id.clone(), 
                                                name: "tool_result".to_string(), 
                                                result: serde_json::Value::String(error_msg) 
                                            });
                                        }
                                    }
                                }
                                Block::ToolResult { .. } => {
                                    // This shouldn't happen in assistant responses
                                    response_blocks.push(block);
                                }
                            }
                        }

                        println!(); // New line after response

                        // Add assistant response to chat
                        if !response_blocks.is_empty() {
                            chat.push(Msg { 
                                role: "assistant", 
                                blocks: response_blocks 
                            });
                        }
                    }
                    Err(fallback_e) => {
                        error!("Both streaming and non-streaming failed: {}", fallback_e);
                        println!("Sorry, I encountered an error. Please try again.");
                    }
                }
            }
        }
    }

    println!("Goodbye! 👋");
    Ok(())
}