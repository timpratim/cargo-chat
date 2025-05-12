# cargo-chat

A fast, modular Rust-based retrieval-augmented generation (RAG) engine for code and text. It provides efficient codebase chunking, embedding, approximate nearest neighbor (ANN) search, and reranking, all via a simple CLI.

## Features

- **Codebase Chunking:** Splits Rust code into semantically meaningful chunks using tree-sitter and code-splitter.
- **Embeddings:** Generates vector embeddings for code/text using [embed_anything] (Jina local embedder).
- **ANN Search:** Fast nearest neighbor search using the [vector] crate (forest-based index).
- **Reranking:** (Stub) Rerank results with a pluggable reranker interface.
- **Serialization:** Indexes are serialized with [bincode] for fast loading and saving.
- **CLI:** Simple, extensible command-line interface with subcommands for indexing and querying.

## Workspace Structure

- `src/` — Main application logic (chunker, embedding, ann, rerank, main.rs)
- `vector/` — Workspace crate providing the ANN index and vector types

## Installation

1. **Clone the repository:**

   ```sh
   git clone <repo-url>
   cd cargo-chat
   ```

2. **Build the project:**

   ```sh
   cargo build --release
   ```

## Usage

### Index a Codebase

```sh
cargo-chat index --repo <PATH_TO_REPO> --out <OUTPUT_DIR> --model <MODEL_DIR>
```

- `--repo <PATH>`: Path to the Rust code repository to index
- `--out <DIR>`: Output directory for the ANN index (index.bin)
- `--model <DIR>`: Directory containing the embedding model (currently uses Jina local embedder)

### Query an Index

```sh
cargo-chat query --index <INDEX_DIR> --model <MODEL_DIR> --rerank <RERANK_MODEL> --q <QUERY> --k <TOP_K>
```

- `--index <DIR>`: Directory containing the ANN index (index.bin)
- `--model <DIR>`: Directory containing the embedding model
- `--rerank <DIR>`: Directory containing the reranker model (currently stubbed)
- `--q <QUERY>`: Query string (text or code)
- `--k <N>`: Number of top results to return

#### Example

```sh
cargo-chat index --repo ./my_rust_project --out ./index --model ./model_dir
cargo-chat query --index ./index --model ./model_dir --rerank ./rerank_model --q "How to implement a binary search?" --k 5
```

## Dependencies

- [clap] — CLI argument parsing
- [anyhow] — Error handling
- [walkdir] — Directory traversal
- [code-splitter] — Code chunking
- [tree-sitter-rust] — Rust syntax parsing
- [vector] — ANN index
- [embed_anything] — Embedding models
- [bincode] — Serialization
- [tokio] — Async runtime

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
[LICENSE.md]: ./LICENSE.md 
