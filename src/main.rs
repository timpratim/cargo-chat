use clap::{Parser, Subcommand};
use anyhow::Result;
use log::info;
use serde_json;
use tracing_subscriber::{fmt, fmt::format::FmtSpan, prelude::*, EnvFilter};
mod chunker; mod embedding; mod ann; mod rerank; mod hyde;
mod openai;
// mod vector;
use ann::ChunkMeta;

#[derive(Parser)]
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer().with_span_events(FmtSpan::CLOSE))
        .with(EnvFilter::from_default_env())
        .init();

    let args = Cli::parse();
    match args.cmd {
        Cmd::Index { repo, out, model } => {
            let embedder = embedding::Embedder::new(Some(model), ())?;
            let chunks = chunker::chunk_repo(&repo)?;
            let mut vecs = Vec::new();
            let mut metas = Vec::new();
            
            info!("Embedding chunks...");
            let app_batch_size = 32; // Configurable batch size

            for chunk_batch in chunks.chunks(app_batch_size) {
                // chunk_batch is a slice of (FilePath: String, CodeSnippet: String)
                let texts_to_embed: Vec<&str> = chunk_batch.iter().map(|(_file_path, code_snippet)| code_snippet.as_str()).collect();
                
                if texts_to_embed.is_empty() {
                    continue;
                }

                let embedding_arrays = embedder.embed_batch(&texts_to_embed, Some(texts_to_embed.len())).await?;

                for (idx, embedding_array) in embedding_arrays.iter().enumerate() {
                    // Retrieve the original file_path and code_snippet for metadata
                    let (file_path, code_snippet) = &chunk_batch[idx];
                    vecs.push(vector::Vector::<512>::from(*embedding_array));
                    metas.push(ChunkMeta { file: file_path.clone(), code: code_snippet.clone() });
                }
            }
            tracing::info!("Embeddings complete. {} embeddings generated.", vecs.len());
            let a = ann::Ann::<512, ChunkMeta>::build(&vecs, &metas);
            {
                let _serialization_span = tracing::info_span!("ann_serialization_and_write").entered();
                std::fs::write(format!("{out}/index.bin"), serde_json::to_vec(&a)?)?;
            }
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