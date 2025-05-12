use clap::{Parser, Subcommand};
use anyhow::Result;
use log::{info, error, warn};
use serde_json;
use tracing_subscriber::{fmt, fmt::format::FmtSpan, prelude::*, EnvFilter};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressStyle};
use futures::future::join_all;

mod chunker; mod embedding; mod ann; mod rerank; mod hyde;
mod openai;
use ann::ChunkMeta;

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
        #[arg(long)]
        model_id: Option<String>,
    },
    /// Query a previously built index using a question to find relevant code and synthesize an answer.
    Query {
        index_dir: String,
        #[arg(long)]
        model_id: Option<String>,
        q: String,
        k: usize,
        #[arg(long)]
        rerank_model: Option<String>,
        #[arg(long, action = clap::ArgAction::SetTrue)]
        use_rerank: bool,
    },
    /// Start an interactive REPL session for efficient iterative indexing and querying.
    Interactive { 
        #[arg(long)]
        model_id: Option<String>,
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
}

struct SessionState {
    embedder: Arc<embedding::Embedder>,
    ann_index: Option<ann::Ann<512, ChunkMeta>>,
    model_id: String,
    current_index_path: Option<String>,
}

impl SessionState {
    fn new( model_id: String, embedder: Arc<embedding::Embedder>) -> Self {
        Self {
            embedder,
            ann_index: None,
            model_id,
            current_index_path: None,
        }
    }
}

async fn execute_index_command(
    embedder: &embedding::Embedder,
    repo_path: &str,
    output_dir: &str,
) -> Result<()> {
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
        let owned_chunk_batch: Vec<(String, String)> = chunk_batch_slice.to_vec();
        let pb_clone = Arc::clone(&pb);
        
        // The embedder reference can be used directly as it's Sync
        // and the tasks are awaited within this function's scope.
        let future = async move {
            let texts_to_embed: Vec<String> = owned_chunk_batch.iter().map(|(_, code_snippet)| code_snippet.clone()).collect();
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
                    for (embedding_idx, embedding_array) in embedding_arrays.iter().enumerate() {
                        let (file_path, code_snippet) = &original_chunk_batch[embedding_idx];
                        vecs.push(vector::Vector::<512>::from(*embedding_array));
                        metas.push(ChunkMeta { file: file_path.clone(), code: code_snippet.clone() });
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
    
    let ann_instance = ann::Ann::<512, ChunkMeta>::build(&vecs, &metas);
    
    std::fs::create_dir_all(output_dir)?;
    let index_file_path = format!("{}/index.bin", output_dir);
    info!("Serializing and writing ANN index to {}", index_file_path);
    std::fs::write(&index_file_path, serde_json::to_vec(&ann_instance)?)?;
    info!("ANN index successfully built and saved to {}", output_dir);
    Ok(())
}

fn print_query_results(hits: &hyde::HydeResponse) {
    // Print the synthesized answer first, without the explicit label
    println!("{}\n", hits.answer.trim());

    // Then print the references, if any
    if !hits.code_refs.is_empty() {
        println!("References:");
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
}

async fn execute_query_command(
    embedder: Arc<embedding::Embedder>,
    ann_index: &ann::Ann<512, ChunkMeta>,
    rerank_model_path: Option<&str>,
    query_string: &str,
    k: usize,
    use_rerank_flag: bool,
) -> Result<()> {
    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set. This key is required for the query command."))?;
    
    let openai_api_url = std::env::var("OPENAI_API_URL").ok();
    
    let openai_client = {
        let client = openai::OpenAIClient::new(&openai_api_key);
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
    
    let hyde_client = openai_client;

    let effective_use_rerank = use_rerank_flag && reranker_instance.is_some();

    let hyde = hyde::Hyde::<512>::new(hyde_client, embedder, ann_index, 1000, reranker_instance);
    
    info!("Retrieving results for query: '{}' with k={}", query_string, k);
    let hits = hyde.retrieve(query_string, k, effective_use_rerank).await?;
    
    print_query_results(&hits);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer().with_span_events(FmtSpan::CLOSE))
        .with(EnvFilter::from_default_env())
        .init();

    let args = Cli::parse();
    match args.cmd {
        Cmd::Index { repo, out,  model_id } => {
            info!("Loading embedder (model_id: {:?})", model_id.as_deref().unwrap_or("default"));
            let embedder = embedding::Embedder::new( model_id)?;
            execute_index_command(&embedder, &repo, &out).await?;
        }
        Cmd::Query { index_dir,  model_id, q, k, rerank_model, use_rerank } => {
            info!("Loading embedder (model_id: {:?})", model_id.as_deref().unwrap_or("default"));
            let embedder = Arc::new(embedding::Embedder::new( model_id)?); 
            
            let index_file_path = format!("{}/index.bin", index_dir);
            info!("Loading ANN index from: {}", index_file_path);
            let bytes = std::fs::read(&index_file_path)?;
            let ann_data: ann::Ann<512, ChunkMeta> = serde_json::from_slice(&bytes)?;
            
            execute_query_command(embedder, &ann_data, rerank_model.as_deref(), &q, k, use_rerank).await?;
        }
        Cmd::Interactive { model_id } => {
            let actual_model_id = model_id.clone().unwrap_or_else(|| "jinaai/jina-embeddings-v2-small-en".to_string());
            info!("Starting interactive session. Loading embedder (model_id: {})", actual_model_id);
            let embedder_instance = embedding::Embedder::new( model_id)?;
            let shared_embedder = Arc::new(embedder_instance);
            let mut session_state = SessionState::new( actual_model_id.clone(), shared_embedder.clone());
            
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

            println!("Interactive Cargo Chat session (Model: {}). Type 'help' for commands, 'exit' to quit.", actual_model_id);

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
                                        match execute_index_command(&session_state.embedder, &args.repo, &args.out).await {
                                            Ok(()) => {
                                                info!("Index command completed successfully. Attempting to load the new index.");
                                                let index_file_path = format!("{}/index.bin", args.out);
                                                match std::fs::read(&index_file_path) {
                                                    Ok(bytes) => match serde_json::from_slice(&bytes) {
                                                        Ok(ann_data) => {
                                                            session_state.ann_index = Some(ann_data);
                                                            session_state.current_index_path = Some(args.out.clone());
                                                            info!("Successfully loaded newly created index from {}", args.out);
                                                        }
                                                        Err(e) => error!("Failed to deserialize newly created index {}: {}", index_file_path, e),
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
                                            Ok(bytes) => match serde_json::from_slice(&bytes) {
                                                Ok(ann_data) => {
                                                    session_state.ann_index = Some(ann_data);
                                                    session_state.current_index_path = Some(args.index_dir.clone());
                                                    info!("Index loaded successfully from {}", args.index_dir);
                                                }
                                                Err(e) => error!("Failed to deserialize index: {}", e),
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
                                            ).await {
                                                error!("Error executing query command: {}", e);
                                            }
                                        } else {
                                            error!("No index loaded. Please use 'load_index <path>' first.");
                                        }
                                    }
                                    ReplSubCmd::Status => {
                                        println!("Session Status:");
                                        println!("  Model ID: {}", session_state.model_id);
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
                                        println!("  query \"<your question>\" [-k <num>] [--use-rerank] [--rerank-model <path>] : Queries the loaded index.");
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