use anyhow::Result;
use code_splitter::{CharCounter, Splitter};
use log::{info, debug, warn};
use ignore::WalkBuilder;
use std::path::Path;
use tree_sitter::Language;
use tree_sitter_python as _;
use tree_sitter_javascript as _;
use tree_sitter_typescript as _;
use tree_sitter_php as _;
use tree_sitter_ruby as _;
use tree_sitter_bash as _;
use tree_sitter_c as _;
use tree_sitter_cpp as _;
use tree_sitter_rust as _;
use tree_sitter_java as _;
use tree_sitter_c_sharp as _;
use tree_sitter_kotlin as _;
use tree_sitter_swift as _;
use tree_sitter_sql as _;
use tree_sitter_go as _;

fn language_for_extension(ext: &str) -> Option<Language> {
    match ext {
        "py" => Some(tree_sitter_python::language()),
        "js" => Some(tree_sitter_javascript::language()),
        "ts" => Some(tree_sitter_typescript::language_typescript()),
        "php" => Some(tree_sitter_php::language()),
        "rb" => Some(tree_sitter_ruby::language()),
        "sh" | "bash" => Some(tree_sitter_bash::language()),
        "c" => Some(tree_sitter_c::language()),
        "cpp" | "cc" | "cxx" | "c++" | "hpp" | "h" => Some(tree_sitter_cpp::language()),
        "rs" => Some(tree_sitter_rust::language()),
        "java" => Some(tree_sitter_java::language()),
        "cs" => Some(tree_sitter_c_sharp::language()),
        "kt" | "kts" => Some(tree_sitter_kotlin::language()),
        "swift" => Some(tree_sitter_swift::language()),
        "sql" => Some(tree_sitter_sql::language()),
        "go" => Some(tree_sitter_go::language()),
        _ => None,
    }
}

#[tracing::instrument(skip(root))]
pub fn chunk_repo(root: &str) -> Result<Vec<(String, String)>> {
    info!("Starting chunking for repo: {}", root);
    // Choose the tree-sitter language dynamically based on file extension.
    // Supported languages include Python, JavaScript, TypeScript, PHP, Ruby,
    // Bash, C, C++, Rust, Java, C#, Kotlin, Swift, SQL and Go.
    let mut out = Vec::new();
    let mut total_files = 0;
    let mut skipped_files = 0;
    let mut total_chunks = 0;
    for entry in WalkBuilder::new(root).standard_filters(true).build().filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e,
            None => {
                debug!("Skipping file with no extension: {}", path.display());
                skipped_files += 1;
                continue;
            }
        };

        let lang = match language_for_extension(ext) {
            Some(l) => l,
            None => {
                debug!("Skipping unsupported file type {}: {}", ext, path.display());
                skipped_files += 1;
                continue;
            }
        };

        let splitter = Splitter::new(lang, CharCounter)
            .expect("Failed to load tree-sitter language")
            .with_max_size(1000);

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
    info!("Chunking complete. Processed {} files, skipped {} files. Total chunks: {}.", total_files, skipped_files, total_chunks);
    Ok(out)
}
