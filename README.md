# Cargo-chat

A Rust-based Retrieval Augmented Generation (RAG) application for code repositories.

## Features

*   **Flexible Embedding Models**: Support for multiple embedding model types including Jina and Qwen3 models with automatic dimension detection (512D for Jina, 1024D for Qwen3).
*   **Code chunking for Rust files**.
*   **Approximate Nearest Neighbor (ANN) index building and querying**.
*   **Hypothetical Document Embeddings (HyDE) for improved retrieval**.
*   **OpenAI integration for answer synthesis** (requires `OPENAI_API_KEY` and optionally `OPENAI_API_URL`).
*   **Optional reranking of search results**.
*   **Interactive REPL mode for efficient iterative operations**.
*   **Detailed logging and tracing capabilities**.

## Prerequisites

*   Rust (latest stable recommended).
*   An embedding model compatible with `embed-anything` (defaults to Jina models, which will be downloaded on first use).
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
# Start the interactive session with default Jina model
RUST_LOG=info ./target/release/cargo_chat interactive

# Start with a specific model type
RUST_LOG=info ./target/release/cargo_chat interactive --model-type qwen3

# Start with a custom model ID
RUST_LOG=info ./target/release/cargo_chat interactive --model-id "custom/jina-model"
```

Once inside the REPL (`cargo-chat (...)> ` prompt), you can use the following commands:

*   `index --repo <path_to_your_code_repo> --out <output_index_dir> [--model-type <jina|qwen3>] [--model-id <custom_model_id>]`
    *   Example: `index --repo ../my_project --out ./my_project_index`
    *   Example with Qwen3: `index --repo ../my_project --out ./my_project_index --model-type qwen3`
    *   Example with custom model: `index --repo ../my_project --out ./my_project_index --model-id "custom/jina-model"`
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
# Example: Index the current directory with default Jina model
RUST_LOG=info ./target/release/cargo_chat index --repo . --out ./output_index

# Example: Index with Qwen3 model
RUST_LOG=info ./target/release/cargo_chat index --repo . --out ./output_index --model-type qwen3

# Example: Index with custom model ID
RUST_LOG=info ./target/release/cargo_chat index --repo . --out ./output_index --model-id "custom/jina-model"
```

*   `--repo <path>`: Path to the code repository to index.
*   `--out <path>`: Path to the directory where the generated index (`index.bin`) will be saved.
*   `--model-type <jina|qwen3>`: Predefined embedding model type (defaults to `jina`).
*   `--model-id <id>`: Custom model ID (overrides `--model-type` if specified).

#### 2. Query an Index

This command queries a previously built ANN index to find relevant code chunks and synthesize an answer.

```bash
# Example: Query an index with default Jina model
RUST_LOG=info ./target/release/cargo_chat query \
    --index_dir ./output_index \
    --q "How do I implement batching for embeddings?" \
    --k 3

# Example: Query with Qwen3 model
RUST_LOG=info ./target/release/cargo_chat query \
    --index_dir ./output_index \
    --model-type qwen3 \
    --q "How do I implement batching for embeddings?" \
    --k 3

# Example: Query with reranking enabled
RUST_LOG=info ./target/release/cargo_chat query \
    --index_dir ./output_index \
    --q "How do I implement batching for embeddings?" \
    --k 3 \
    --use-rerank \
    --rerank_model ./path_to_reranker_model
```

*   `--index_dir <path>`: Path to the directory containing the `index.bin` file.
*   `--model-type <jina|qwen3>`: Predefined embedding model type (defaults to `jina`).
*   `--model-id <id>`: Custom model ID (overrides `--model-type` if specified).
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

## Embedding Models

Cargo-chat supports multiple embedding model types with automatic dimension detection:

### Supported Model Types

- **Jina Models** (512-dimensional embeddings)
  - Default: `jinaai/jina-embeddings-v2-small-en`
  - Compatible with any Jina embedding model via `embed-anything`
  
- **Qwen3 Models** (1024-dimensional embeddings)
  - Default: `Qwen/Qwen3-Embedding-0.6B`
  - Compatible with Qwen3 embedding models

### Model Selection

You can specify models in several ways:

1. **Predefined Types**: Use `--model-type jina` or `--model-type qwen3`
2. **Custom Model IDs**: Use `--model-id "your/custom/model"` (auto-detects type based on model name)
3. **Default**: If no model is specified, defaults to Jina model

### Model Compatibility

- Models are automatically downloaded on first use via `embed-anything`
- The application automatically detects embedding dimensions (512D for Jina, 1024D for Qwen3)
- Index files are compatible across different model types as long as dimensions match

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
    - ✅ **Completed**: Implemented flexible embedding model selection with enum-based API
    - ✅ **Completed**: Added support for Jina and Qwen3 models with automatic dimension detection
    - Explore and evaluate additional embedding models (e.g., E5, GTE, BGE) for improved performance and domain-specific relevance
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