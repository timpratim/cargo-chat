# Cargo Chat 🦀💬

**AI-powered code search and Q&A for your repositories**

Cargo Chat helps you understand and navigate codebases using natural language. Ask questions about your code and get intelligent answers backed by semantic search and AI analysis.

## ✨ What Can You Do?

- **Ask questions about your codebase** in plain English
- **Search across multiple programming languages** (Rust, JavaScript, TypeScript, Java, C++, Python, and more)
- **Get context-aware answers** that understand your specific code
- **Interactive chat mode** for exploring codebases efficiently
- **Smart code retrieval** that finds relevant code snippets automatically

## 🚀 Quick Start

### Prerequisites

- **Rust** (latest stable version)
- **OpenAI API Key** (set as `OPENAI_API_KEY` environment variable)
  - Get one at [OpenAI's website](https://platform.openai.com/api-keys)
- That's it! Embedding models download automatically on first use.

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cargo-chat

# Build the application
cargo build --release

# Optional: Enable command history (saves your commands)
cargo build --release --features with-file-history
```

## 💡 How to Use

### Step 1: Start Interactive Mode

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Start cargo-chat
./target/release/cargo_chat interactive
```

### Step 2: Index Your Codebase

```bash
# Inside cargo-chat, index your project
index --repo /path/to/your/project --out ./my_project_index
```

### Step 3: Ask Questions!

```bash
# Ask about your code
query "How does authentication work in this codebase?"
query "Show me examples of error handling"
query "What are the main API endpoints?"
```

### Available Commands

- `index --repo <path> --out <index_dir>` - Index a codebase
- `load_index <index_dir>` - Load an existing index
- `query "<question>"` - Ask questions about your code
- `status` - Show current model and index info
- `help` - Show all commands
- `exit` - Quit cargo-chat

## 🔧 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Make sure you've set the `OPENAI_API_KEY` environment variable
- Verify your API key is valid and has sufficient credits

**"Model download failed"**
- Check your internet connection
- Ensure you have sufficient disk space for embedding models

**"No results found"**
- Try rephrasing your question
- Make sure your codebase was indexed successfully
- Check if your question matches the programming language in your codebase

### Enable Debug Logging

For troubleshooting, you can enable detailed logging:

```bash
# Show debug information
RUST_LOG=debug ./target/release/cargo_chat interactive
```

## 🤝 Contributing

We welcome contributions to Cargo Chat! Here's how you can help:

- **Report bugs** by opening an issue
- **Suggest features** or improvements
- **Submit pull requests** with bug fixes or new features
- **Improve documentation** to help other users

Before contributing:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

See [LICENSE.md](./LICENSE.md) for license information.

---

**Happy coding! 🚀**
- **Java** (.java) - Standard Java language support
- **C++** (.cpp, .cxx, .cc, .hpp, .hxx, .hh) - Modern C++ support
- **C** (.c, .h) - Standard C language support
- **Ruby** (.rb) - Ruby language support
- **C#** (.cs) - C# language support
- **Swift** (.swift) - Swift language support
- **Go** (.go) - Go language support
- **Python** (.py, .pyx, .pyi) - Python 3+ support including type hints
- **Markdown** (.md, .markdown) - Documentation and README files

### Language Detection Features

- **Automatic Detection**: Files are automatically categorized by extension
- **Language-Specific Parsing**: Each language uses its dedicated tree-sitter grammar for accurate code chunking
- **Adaptive AI Responses**: Prompts and responses are tailored to the detected programming language