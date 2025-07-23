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
mod openai; mod language;
use ann::ChunkMeta;
use embedding::EmbeddingModel;

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

    let hyde = hyde::Hyde::new(hyde_client, answer_client, embedder, ann_index, 1000, reranker_instance);
    
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
            
            execute_query_command(embedder, &ann_data, rerank_model.as_deref(), &q, k, use_rerank, &hyde_model, &answer_model).await?;
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
                                            if let Err(e) = execute_query_command(
                                                session_state.embedder.clone(),
                                                ann_data,
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
    }
    Ok(())
}