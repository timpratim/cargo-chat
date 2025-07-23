use anyhow::Result;
use code_splitter::{CharCounter, Splitter};
use log::{info, debug, warn};
use ignore::WalkBuilder;
use crate::language::{detect_language_from_extension, get_all_supported_extensions, SupportedLanguage};
use std::collections::HashMap;

/// Represents a code chunk extracted from a repository file
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// The file path where this chunk originated
    pub file_path: String,
    /// The actual code content of the chunk
    pub content: String,
    /// The detected programming language (if any)
    pub language: Option<String>,
    /// The file extension (if any)
    pub extension: Option<String>,
}

#[tracing::instrument(skip(root))]
pub fn chunk_repo(root: &str) -> Result<Vec<CodeChunk>> {
    info!("Starting chunking for repo: {}", root);
    
    // Create splitters for each supported language
    let mut splitters: HashMap<SupportedLanguage, Splitter<CharCounter>> = HashMap::new();
    
    // Get all supported extensions for filtering
    let supported_extensions = get_all_supported_extensions();
    let mut out = Vec::new();
    let mut total_files = 0;
    let mut skipped_files = 0;
    let mut total_chunks = 0;
    let mut files_by_language: HashMap<SupportedLanguage, usize> = HashMap::new();
    
    for entry in WalkBuilder::new(root).standard_filters(true).build().filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        
        // Detect language from file extension
        let extension = path.extension().and_then(|e| e.to_str()).map(|s| s.to_string());
        let language = match extension.as_ref() {
            Some(ext) => {
                if supported_extensions.contains(&ext.as_str()) {
                    detect_language_from_extension(ext)
                } else {
                    None
                }
            }
            None => None,
        };
        
        let language = match language {
            Some(lang) => lang,
            None => {
                debug!("Skipping unsupported file: {}", path.display());
                skipped_files += 1;
                continue;
            }
        };
        // Get or create splitter for this language
        let splitter = match splitters.get(&language) {
            Some(s) => s,
            None => {
                let tree_sitter_lang = language.tree_sitter_language();
                let new_splitter = Splitter::new(tree_sitter_lang, CharCounter)
                    .map_err(|e| anyhow::anyhow!("Failed to load tree-sitter language for {}: {}", language.display_name(), e))?
                    .with_max_size(1000);
                splitters.insert(language.clone(), new_splitter);
                splitters.get(&language).unwrap()
            }
        };
        
        total_files += 1;
        *files_by_language.entry(language.clone()).or_insert(0) += 1;
        info!("Processing {} file: {}", language.display_name(), path.display());
        
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
                warn!("Failed to split {} file {}: {}", language.display_name(), path.display(), e);
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
            out.push(CodeChunk {
                file_path: path.display().to_string(),
                content: snippet,
                language: Some(language.display_name().to_string()),
                extension: extension.clone(),
            });
            file_chunk_count += 1;
        }
        total_chunks += file_chunk_count;
        info!("File {}: {} chunks generated", path.display(), file_chunk_count);
    }
    
    // Log summary by language
    info!("Chunking complete. Total files processed: {}, skipped: {}, total chunks: {}", total_files, skipped_files, total_chunks);
    for (lang, count) in files_by_language {
        info!("  {}: {} files", lang.display_name(), count);
    }
    
    Ok(out)
}
