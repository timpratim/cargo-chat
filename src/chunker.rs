use anyhow::Result;
use code_splitter::{CharCounter, Splitter};
use log::{info, debug, warn};
use ignore::WalkBuilder;

#[tracing::instrument(skip(root))]
pub fn chunk_repo(root: &str) -> Result<Vec<(String, String)>> {
    info!("Starting chunking for repo: {}", root);
    let lang = tree_sitter_rust::language();
    let splitter = Splitter::new(lang, CharCounter)
        .expect("Failed to load tree-sitter language")
        .with_max_size(1000);
    let mut out = Vec::new();
    let mut total_files = 0;
    let mut skipped_files = 0;
    let mut total_chunks = 0;
    for entry in WalkBuilder::new(root).standard_filters(true).build().filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !path.extension().map_or(false, |e| e == "rs") {
            debug!("Skipping non-Rust file: {}", path.display());
            skipped_files += 1;
            continue;
        }
        total_files += 1;
        info!("Processing file: {}", path.display());
        let code = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to read file {}: {}", path.display(), e);
                skipped_files += 1;
                continue;
            }
        };
        let code_bytes = code.as_bytes();
        let chunks = match splitter.split(code_bytes) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to split file {}: {}", path.display(), e);
                skipped_files += 1;
                continue;
            }
        };
        let mut file_chunk_count = 0;
        for chunk in chunks {
            let start = chunk.range.start_byte;
            let end = chunk.range.end_byte;
            let snippet = match std::str::from_utf8(&code_bytes[start..end]) {
                Ok(s) => s.to_string(),
                Err(e) => {
                    warn!("Invalid UTF-8 in file {}: {}", path.display(), e);
                    continue;
                }
            };
            out.push((path.display().to_string(), snippet));
            file_chunk_count += 1;
        }
        total_chunks += file_chunk_count;
        info!("File {}: {} chunks generated", path.display(), file_chunk_count);
    }
    info!("Chunking complete. Processed {} Rust files, skipped {} files. Total chunks: {}.", total_files, skipped_files, total_chunks);
    Ok(out)
}
