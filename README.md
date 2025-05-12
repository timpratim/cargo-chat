# Cargo-chat

A Rust-based Retrieval Augmented Generation (RAG) application for code repositories.

## Features

*   Local embedding model loading (currently Jina Embeddings v2 Small EN via `embed-anything`).
*   Code chunking for Rust files.
*   Approximate Nearest Neighbor (ANN) index building and querying.
*   Hypothetical Document Embeddings (HyDE) for improved retrieval.
*   OpenAI integration for answer synthesis (requires `OPENAI_API_KEY` and optionally `OPENAI_API_URL`).
*   Optional reranking of search results.
*   Interactive REPL mode for efficient iterative operations.
*   Detailed logging and tracing capabilities.

## Prerequisites

*   Rust (latest stable recommended).
*   An embedding model compatible with `embed-anything` (the default is `jinaai/jina-embeddings-v2-small-en`, which will be downloaded on first use).
*   Optionally, an OpenAI API key set as an environment variable (`OPENAI_API_KEY`) for answer synthesis and advanced HyDE features.
*   ONNX Runtime and its dependencies if you plan to use the reranking feature. Installation instructions can be found on the [ONNX Runtime website](https://onnxruntime.ai/docs/install/).

## Building

```bash
cargo build --release
```

### Compiling with REPL Command History

To enable persistent command history for the interactive REPL mode (saving commands to `~/.cargo_chat_history`), compile the application with the `with-file-history` feature:

```bash
cargo build --features with-file-history
```

For a release build with this feature:

```bash
cargo build --release --features with-file-history
```

## Usage

`cargo-chat` offers both one-shot commands and an interactive REPL mode.

### Interactive Mode (Recommended for multiple operations)

The interactive mode loads the embedding model once at the start, making subsequent indexing and querying much faster.

```bash
# Start the interactive session, specifying the directory containing your embedding model
# (If the model doesn't exist locally, embed-anything will attempt to download it here)
RUST_LOG=info ./target/release/cargo_chat interactive --model_dir ./embedding_model_cache
```

Once inside the REPL (`cargo-chat (...)> ` prompt), you can use the following commands:

*   `index --repo <path_to_your_code_repo> --out <output_index_dir>`
    *   Example: `index --repo ../my_project --out ./my_project_index`
*   `load_index <path_to_existing_index_dir>`
    *   Example: `load_index ./my_project_index`
*   `query "<your question>" [-k <num_results>] [--use-rerank] [--rerank-model <path_to_rerank_model>]`
    *   Example: `query "how is the ann index built?" -k 3`
    *   Example with reranking: `query "explain batching" -k 5 --use-rerank --rerank-model ./rerank_model_dir`
*   `status`: Shows the currently loaded model and index information.
*   `help`: Displays help for REPL commands.
*   `exit`: Exits the interactive session.

### One-Shot Commands

#### 1. Index a Repository

This command chunks the specified repository, generates embeddings for the code chunks, and builds an ANN index.

```bash
# Example: Index the current directory, save the index to ./output_index
RUST_LOG=info ./target/release/cargo_chat index --repo . --out ./output_index
```

*   `--repo <path>`: Path to the code repository to index.
*   `--out <path>`: Path to the directory where the generated index (`index.bin`) will be saved.
*   `--model_dir <path>`: Path to the directory containing the embedding model. If the model is not present, `embed-anything` might attempt to download it here.

#### 2. Query an Index

This command queries a previously built ANN index to find relevant code chunks and synthesize an answer.

```bash
# Example: Query an index located in ./output_index,
# and optionally specify a reranker model.
RUST_LOG=info ./target/release/cargo_chat query \
    --index_dir ./output_index \
    --q "How do I implement batching for embeddings?" \
    --k 3 \\
    # --use-rerank \\
    # --rerank_model ./path_to_reranker_model
```

*   `--index_dir <path>`: Path to the directory containing the `index.bin` file.
*   `--q "<query_string>"`: The question you want to ask.
*   `--k <num>`: The number of top results to retrieve.
*   `--rerank_model <path>`: (Optional) Path to a reranking model directory.
*   `--use-rerank`: (Optional) Flag to enable reranking if a `rerank_model` is provided.

## Logging and Tracing

The application uses the `tracing` framework for structured logging. You can control the log verbosity using the `RUST_LOG` environment variable.

**Examples:**

*   Show general informational messages:
    ```bash
    RUST_LOG=info ./target/release/cargo_chat <command> <args>
    ```
*   Show debug messages from `cargo_chat` modules and info from others:
    ```bash
    RUST_LOG=cargo_chat=debug,info ./target/release/cargo_chat <command> <args>
    ```
*   Show trace level (very verbose) messages for all `cargo_chat` modules, and info for other crates:
    ```bash
    RUST_LOG=info,cargo_chat=trace ./target/release/cargo_chat <command> <args>
    ```
*   Show trace level messages only for the `cargo_chat::embedding` module:
    ```bash
    RUST_LOG=cargo_chat::embedding=trace ./target/release/cargo_chat <command> <args>
    ```

The logs include timestamps and span events, which can show the duration of specific operations (e.g., `close time.busy=... time.idle=...`). This is helpful for performance analysis.

## Environment Variables

*   `OPENAI_API_KEY`: Your OpenAI API key. Required for answer synthesis and some HyDE features.
*   `OPENAI_API_URL`: (Optional) Custom base URL for the OpenAI API (e.g., for local LLM proxies).
*   `RUST_LOG`: Controls logging verbosity (see Tracing section).

## Workspace Structure
- `src/` — Main application logic (chunker, embedding, ann, rerank, main.rs)

## Dependencies
- [clap] — CLI argument parsing
- [anyhow] — Error handling
- [walkdir] — Directory traversal
- [code-splitter] — Code chunking
- [tree-sitter-rust] — Rust syntax parsing
- [embed_anything] — Embedding models
- [bincode] — Serialization
- [tokio] — Async runtime
- [rustyline] — REPL line editing

## Todos

- **Embedding Model Enhancement:**
    - Explore and evaluate alternative embedding models (e.g., E5, GTE, BGE) for improved performance and domain-specific relevance.
    - Implement configurability for the embedding model selection.
    - Make embedding dimensions dynamic, adapting to the chosen model's output.
- **Multi-Language Support for Code Chunking:**
    - Integrate dynamic loading of tree-sitter grammars to support various programming languages beyond Rust.
    - Update file filtering logic to accommodate multiple file extensions based on supported languages.
    - Investigate and implement language-specific chunking parameters for optimal code segmentation.
- **Configuration Management:**
    - Introduce a configuration file (e.g., TOML, YAML) for managing settings like model paths, chunking parameters, and API keys, reducing reliance on command-line arguments and environment variables.
- **Advanced Reranking:**
    - Explore and integrate more sophisticated reranking models and strategies.
- **Error Handling and Resilience:**
    - Enhance error reporting with more context-specific details.
    - Improve resilience to failures during long-running processes like indexing.
- **Testing:**
    - Expand unit and integration test coverage, particularly for embedding, chunking, and indexing logic.
- **Documentation:**
    - Provide more detailed documentation for developers, including architecture overview and module-specific guides.

## Contribution
Contributions are welcome! Please open issues or pull requests. All contributions are licensed under the terms in [LICENSE.md].

## License
See [LICENSE.md].

[clap]: https://crates.io/crates/clap
[anyhow]: https://crates.io/crates/anyhow
[walkdir]: https://crates.io/crates/walkdir
[code-splitter]: https://crates.io/crates/code-splitter
[tree-sitter-rust]: https://crates.io/crates/tree-sitter-rust
[vector]: https://crates.io/crates/vector
[embed_anything]: https://crates.io/crates/embed_anything
[bincode]: https://crates.io/crates/bincode
[tokio]: https://crates.io/crates/tokio
[rustyline]: https://crates.io/crates/rustyline
[LICENSE.md]: ./LICENSE.md 