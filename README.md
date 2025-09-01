# ChatGPT Complex Networks

A toolkit for generating, analyzing, and visualizing complex networks from conversational data. This project provides tools for embedding conversations, generating similarity-based edges, and performing network analysis on conversation graphs.

## Features

- **Plugin-Based Data Integration**:
  - Extensible integration system for various data sources
  - Built-in support for chat logs (JSON, Markdown), bookmarks, playlists
  - Easy to add custom integrations

- **Flexible Embedding Generation**:
  - Multiple LLM providers: Ollama (local), OpenAI, Anthropic/Voyage, HuggingFace, Cohere
  - Memory-efficient TF-IDF for large datasets
  - Advanced chunking strategies (sliding window, sentence, paragraph)
  - Role-based weighting for conversations
  - Configurable aggregation methods (mean, weighted, PCA, max-pool)

- **Graph Generation**:
  - Generate edges based on cosine similarity between node embeddings
  - GPU-accelerated edge generation for large networks
  - Filter edges using similarity cutoffs
  - Normalize and rescale edge weights

- **Network Analysis**:
  - Compute comprehensive graph statistics (centrality, clustering, etc.)
  - Export to common graph formats (GEXF, GraphML, GML)
  - Analyze conversation connectivity and important nodes

- **Recommendation System**:
  - Find semantically similar conversations
  - Interactive REPL for exploring the conversation graph
  - Adjustable weighting between similarity and centrality

## Installation

```bash
# Clone the repository
git clone https://github.com/queelius/llm-semantic-net.git
cd llm-semantic-net

# Install as package
pip install -e .

# For development (includes testing tools)
pip install -e ".[dev]"
# Or:
pip install -r requirements-dev.txt
```

After installation, the following commands become available:
- `semnet` - Main CLI for integration-based semantic network generation
- `semnet-rec` - Recommendation system

## Usage

### Quick Start

```bash
# List available integrations
semnet list

# Import your data (e.g., JSON chat logs)
semnet import --integration chatlog.json \
    --source ./data/conversations --output-dir ./data/imported

# Generate embeddings (with separate storage)
semnet embed --input-dir ./data/imported \
    --output-dir ./data/embeddings \
    --integration chatlog --method llm

# Create edges and export
semnet edges --input-dir ./data/embeddings \
    --output-file edges.json --threshold 0.7
    
semnet export --nodes-dir ./data/imported \
    --edges-file edges.json --format gexf --output-file graph.gexf
```

### Advanced Embedding Options

```bash
# Use OpenAI embeddings
semnet embed --input-dir ./data/imported \
    --output-dir ./data/embeddings \
    --integration chatlog --method llm \
    --llm-provider openai \
    --llm-config '{"model": "text-embedding-3-large"}'

# Chunked embeddings with role weighting
semnet embed --input-dir ./data/imported \
    --output-dir ./data/embeddings \
    --integration chatlog --method llm \
    --chunk-size 1024 --chunk-overlap 100 \
    --aggregation-method weighted_mean \
    --role-weights '{"user": 1.5, "assistant": 1.0}'
```

### GPU-Accelerated Edge Generation

For large datasets, you can use GPU acceleration for edge generation:

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then use the GPU edge generation module
python graph/gpu-edge.py --input-dir ./data/embeddings \
    --output-file edges_gpu.json
```

### Recommendation System

```bash
# Start interactive recommendation shell
python rec-conv.py --nodes-dir ./data/embeddings_json \
    --csv nodes.csv --repl

# One-shot recommendation
python rec-conv.py --nodes-dir ./data/embeddings_json \
    --csv nodes.csv --recommend new_conv.json --topk 8
```

## Project Structure

```
├── cli.py                     # Main CLI with integration support
├── rec_conv.py                # Recommendation system
├── integrations/             # Plugin-based data integrations
│   ├── base.py              # Abstract base classes
│   ├── chatlog/             # Chat log integrations
│   ├── bookmarks/           # Bookmark integrations
│   └── playlist/            # Playlist integrations
├── embedding/                # Embedding generation
│   ├── llm_providers.py     # Multiple LLM provider support
│   ├── chunking.py          # Advanced chunking strategies
│   ├── tfidf_memory_efficient.py  # Memory-efficient TF-IDF
│   └── llm_embedding_model.py     # Legacy Ollama support
├── graph/                    # Graph processing
│   ├── edge_utils.py        # Edge generation
│   ├── export_utils.py      # Export to various formats
│   └── gpu-edge.py          # GPU acceleration
├── tests/                    # Test suite
│   ├── test_core/           # Core functionality tests
│   ├── test_embedding/      # Embedding tests
│   ├── test_integrations/   # Integration tests
│   └── conftest.py          # Shared fixtures
└── config.example.yaml       # Configuration examples
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only

# Generate HTML coverage report
pytest --cov --cov-report=html
```

## Configuration

### Environment Variables

Set these environment variables for different LLM providers:

```bash
# Ollama (local)
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="nomic-embed-text"

# OpenAI
export OPENAI_API_KEY="sk-..."

# Other providers
export VOYAGE_API_KEY="..."        # Anthropic/Voyage
export HUGGINGFACE_API_KEY="..."
export COHERE_API_KEY="..."
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```
@software{chatgpt-complex-net,
  author = {Alex Towell},
  title = {ChatGPT Complex Networks},
  year = {2025},
  url = {https://github.com/queelius/chatgpt-complex-net}
}
```