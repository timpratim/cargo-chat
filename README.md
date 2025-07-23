# Cargo Chat 🦀💬

**AI-powered code search and Q&A for your repositories**

Cargo Chat helps you understand and navigate codebases using natural language. Ask questions about your code and get intelligent answers backed by semantic search and AI analysis.

## ✨ What Can You Do?

- **Ask questions about your codebase** in plain English
- **Search across multiple programming languages** (Rust, JavaScript, TypeScript, Java, C++, Python, and more)
- **Get context-aware answers** that understand your specific code
- **Interactive chat mode** for exploring codebases efficiently
- **Smart code retrieval** that finds relevant code snippets automatically

## 🧠 Intent-Aware Search

Cargo Chat features an advanced **intent-aware search system** that understands what you're looking for and adapts its responses accordingly:

### 🎯 Smart Query Classification

- **Automatic Intent Detection**: Recognizes whether you want implementation details, explanations, debugging help, or architectural overviews
- **Language-Aware Filtering**: Detects target programming languages from your queries and prioritizes relevant files
- **Confidence-Based Results**: Uses AI confidence scoring to determine the best mix of code and documentation

### 📁 Advanced Filtering Capabilities

- **Folder-Specific Search**: `"search in src folder"` or `"look in tests/"` to target specific directories
- **Extension-Based Filtering**: `"only .rs files"` or `"show me Python code"` to focus on specific file types
- **Exclusion Patterns**: `"exclude tests"` or `"skip target folder"` to remove unwanted results
- **Combined Filtering**: `"Rust files in src folder excluding tests"` for precise targeting

### 🤖 Intelligent Response Generation

- **Role-Based Prompts**: System adapts as a code expert, senior developer, technical educator, or debugging specialist based on your query
- **Intent-Specific Responses**:
  - **"How does X work?"** → Architectural explanations with code examples
  - **"Show me the implementation of Y"** → Direct code focus with minimal documentation
  - **"Debug this error"** → Problem-solving approach with relevant code context
  - **"Explain this concept"** → Educational responses with theory and practice

### 💡 Example Queries

```bash
# Intent: Implementation focus
query "How does the Hyde algorithm work in this Rust codebase?"

# Intent: Folder-specific search
query "Show me error handling patterns in the src folder"

# Intent: Language and exclusion filtering
query "Find authentication code in Python files, exclude tests"

# Intent: Debugging assistance
query "Why might this function be returning None?"
```

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