use clap::{Parser, Subcommand}; // Command line argument parsing
use anyhow::Result; // Error handling
use log::info; // Logging
use serde_json; // JSON serialization
mod chunker; // Chunking
mod embedding; // Embedding
mod ann; // ANN
mod rerank; // Reranking
mod hyde; // HyDE
mod openai; // OpenAI
// mod vector;
use ann::ChunkMeta;

#[derive(Parser)] // Command line argument parsing
struct Cli { 
    #[command(subcommand)]
    cmd: Cmd,
}
#[derive(Subcommand)]
enum Cmd {
    Index { repo: String, out: String, model: String },
    Query {
        index: String,
        model: String,
        rerank: String,
        q: String,
        k: usize,
        #[arg(long, default_value_t = true)]
        no_rerank: bool,
    },
}

#[tokio::main] // Async main function
async fn main() -> Result<()> {
    env_logger::init();
    let args = Cli::parse();
    match args.cmd {
        Cmd::Index { repo, out, model } => {
            let embedder = embedding::Embedder::new(Some(model), ())?;
            let chunks = chunker::chunk_repo(&repo)?;
            let mut vecs = Vec::new();
            let mut metas = Vec::new();
            //log embedding
            info!("Embedding chunks...");
            for (_i, (file_path, code_snippet)) in chunks.iter().enumerate() {
                let v = embedder.embed(code_snippet).await?;
                vecs.push(vector::Vector::<512>::from(v));
                metas.push(ChunkMeta { file: file_path.clone(), code: code_snippet.clone() });
            }
            let a = ann::Ann::<512, ChunkMeta>::build(&vecs, &metas);
            std::fs::write(format!("{out}/index.bin"), serde_json::to_vec(&a)?)?;
        }
        Cmd::Query { index, model, rerank, q, k, no_rerank } => {
            let embedder = embedding::Embedder::new(Some(model.clone()), ())?;
            let bytes   = std::fs::read(index.clone() + "/index.bin")?;
            let ann: ann::Ann<512, ChunkMeta> = serde_json::from_slice(&bytes)?;
            let openai_api_key = std::env::var("OPENAI_API_KEY").ok();
            let openai_api_url = std::env::var("OPENAI_API_URL").ok();
            let openai_client = openai_api_key.as_ref().map(|key| {
                let client = openai::OpenAIClient::new(key);
                if let Some(url) = openai_api_url.as_ref() {
                    client.with_api_url(url)
                } else {
                    client
                }
            });
            let reranker = if !no_rerank {
                Some(rerank::Reranker::new(&rerank, ())?)
            } else {
                None
            };
            let hyde = if let Some(client) = openai_client {
                hyde::Hyde::<512>::new(client, embedder, &ann, 1000, reranker)
            } else {
                hyde::Hyde::<512>::new(openai::OpenAIClient::new(""), embedder, &ann, 1000, reranker)
            };
            let hits = hyde.retrieve(&q, k, !no_rerank).await?;

            for (rank, res) in hits.code_refs.iter().enumerate() {
                let file = &res.meta.file;
                let code = &res.meta.code;
                let snippet = code.trim();
                let snippet = if snippet.len() > 200 {
                    format!("{}...", &snippet[..200])
                } else {
                    snippet.to_string()
                };
                println!(
                    "Result {}:\n  File: {}\n  Chunk ID: {}\n  Distance: {:.4}\n  Code:\n{}\n",
                    rank + 1,
                    file,
                    res.index,
                    res.distance,
                    snippet
                );
            }
            // Print the synthesized answer
            println!("\nSYNTHESIZED ANSWER:\n{}\n", hits.answer);
        }
    }
    Ok(())
}
