# Makefile for rag-rs project

# Default target
.PHONY: all
all: build

# Build the project in release mode with file history support
.PHONY: build
build:
	@echo "Building rag-rs in release mode..."
	cargo build --release --features with-file-history

# Build the project in debug mode
.PHONY: build-debug
build-debug:
	@echo "Building rag-rs in debug mode..."
	cargo build

# Run the project (interactive mode) using the release build
.PHONY: run
run:
	@echo "Running rag-rs in interactive mode..."
	cargo run --release --features with-file-history -- interactive

# Run the project with specific arguments using the release build
.PHONY: run-args
run-args:
	@echo "Usage: make run-args ARGS=\"your arguments here\""
	@if [ -n "$(ARGS)" ]; then \
		cargo run --release --features with-file-history -- $(ARGS); \
	else \
		echo "No arguments provided. Use: make run-args ARGS=\"your arguments\""; \
	fi

# Clean the project
.PHONY: clean
clean:
	@echo "Cleaning project..."
	cargo clean

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	cargo test

# Format code
.PHONY: fmt
fmt:
	@echo "Formatting code..."
	cargo fmt

# Check code formatting
.PHONY: fmt-check
fmt-check:
	@echo "Checking code formatting..."
	cargo fmt -- --check

# Run clippy lints
.PHONY: lint
lint:
	@echo "Running clippy lints..."
	cargo clippy -- -D warnings

# Help command
.PHONY: help
help:
	@echo "cargo-chat Makefile commands:"
	@echo "  make build       - Build the project in release mode"
	@echo "  make build-debug - Build the project in debug mode"
	@echo "  make run         - Run the project in interactive mode"
	@echo "  make run-args    - Run with custom arguments, e.g., make run-args ARGS=\"query index_dir 'query' 3\""
	@echo "  make clean       - Clean the project"
	@echo "  make test        - Run tests"
	@echo "  make fmt         - Format code"
	@echo "  make fmt-check   - Check code formatting"
	@echo "  make lint        - Run clippy lints"
	@echo "  make help        - Show this help message"
