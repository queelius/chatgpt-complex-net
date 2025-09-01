# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# For development with testing tools
pip install -e ".[dev]"
# Or:
pip install -r requirements-dev.txt

# After installation, these commands are available:
# semnet - main CLI for integration-based semantic network generation
# semnet-rec - recommendation system
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests

# Run tests for specific module
pytest tests/test_embedding/
pytest tests/test_integrations/

# Run with verbose output
pytest -vv

# Generate HTML coverage report
pytest --cov --cov-report=html
# Open htmlcov/index.html in browser
```

### Core Workflow Commands

#### 1. List Available Integrations
```bash
semnet list
```

#### 2. Import Data
```bash
# Import JSON chat logs
semnet import --integration chatlog.json \
    --source ./data/conversations --output-dir ./data/imported

# Import markdown chat logs
semnet import --integration chatlog.markdown \
    --source ./conversations --output-dir ./data/imported \
    --config '{"link_strategy": "temporal"}' --extract-edges

# Import Chrome bookmarks
semnet import --integration bookmarks.chrome \
    --source ~/Library/Application\ Support/Google/Chrome/Default/Bookmarks \
    --output-dir ./data/bookmarks --extract-edges
```

#### 3. Generate Embeddings
```bash
# LLM embeddings (default: Ollama)
semnet embed --input-dir ./data/imported \
    --output-dir ./data/embeddings \
    --integration chatlog --method llm

# LLM embeddings with different providers
semnet embed --input-dir ./data/imported \
    --output-dir ./data/embeddings \
    --integration chatlog --method llm \
    --llm-provider openai \
    --llm-config '{"model": "text-embedding-3-large"}'

# Chunked embeddings for long documents
semnet embed --input-dir ./data/imported \
    --output-dir ./data/embeddings \
    --integration chatlog --method llm \
    --chunk-size 1024 --chunk-overlap 100 \
    --chunk-strategy sentence \
    --aggregation-method weighted_mean \
    --role-weights '{"user": 1.5, "assistant": 1.0}'

# Memory-efficient TF-IDF embeddings
semnet embed --input-dir ./data/imported \
    --output-dir ./data/embeddings \
    --integration chatlog --method tfidf \
    --max-features 5000 --batch-size 100
```

#### 4. Generate Graph Edges
```bash
# Create similarity edges from embeddings
semnet edges --input-dir ./data/embeddings \
    --output-file edges.json --threshold 0.7

# For GPU acceleration, use the standalone script:
python graph/gpu-edge.py --input-dir ./data/embeddings \
    --output-file edges_gpu.json
```

#### 5. Export for Visualization
```bash
# Export to various graph formats
semnet export --nodes-dir ./data/imported \
    --edges-file edges.json --format gexf --output-file graph.gexf

# Supported formats: gexf, graphml, gml
```

#### 6. Recommendation System
```bash
# Interactive REPL
semnet-rec --nodes-dir ./data/embeddings \
    --csv nodes.csv --repl

# One-shot recommendation
semnet-rec --nodes-dir ./data/embeddings \
    --csv nodes.csv --recommend new_conv.json --topk 8
```

## Architecture

### Core Components

**Integration System**:
- Plugin-based architecture for extensible data source support
- Built-in integrations: chat logs (JSON/Markdown), bookmarks, playlists
- Easy to add custom integrations by implementing the DataSource interface

**Graph Processing Pipeline**:
1. **Data Import** (`integrations/`): Load data from various sources via plugins
2. **Embedding Generation** (`embedding/`): Convert content to vector embeddings
   - Multiple LLM providers (Ollama, OpenAI, Anthropic, HuggingFace, Cohere)
   - Classical TF-IDF with memory-efficient implementation
   - Advanced chunking and aggregation strategies
3. **Edge Generation** (`graph/edge_utils.py`, `graph/gpu-edge.py`): Create weighted edges based on similarity
4. **Graph Export** (`graph/export_utils.py`): Export to standard formats for visualization
5. **Recommendation Engine** (`rec-conv.py`): Find similar items using similarity and centrality

### Key Design Patterns

**Embedding Storage**: 
- Embeddings stored as separate JSON files in dedicated directory
- Each file contains: source_file reference, node_id, embedding vector, metadata
- Supports multiple embedding types and providers
- Includes chunking metadata when applicable

**Integration Interface**:
```python
class DataSource(ABC):
    def load(self, source: str) -> None
    def extract_nodes(self) -> Iterator[Node]
    def extract_edges(self, nodes: Optional[List[Node]]) -> Iterator[Edge]
    def get_node_content_for_embedding(self, node: Node) -> str
```

**GPU Acceleration**: The `gpu-edge.py` module uses PyTorch for batch cosine similarity computation on CUDA devices when available.

**Normalization**: All embeddings are L2-normalized before similarity computation to ensure cosine similarity equals dot product.

### Configuration

**LLM Provider Settings**:
Multiple embedding providers are supported via environment variables or CLI configuration:

- **Ollama** (default, local):
  - `OLLAMA_HOST`: Default "http://localhost:11434"
  - `OLLAMA_MODEL`: Default "nomic-embed-text"
  
- **OpenAI**:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - Models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
  
- **Anthropic/Voyage**:
  - `VOYAGE_API_KEY`: Your Voyage AI API key
  - Models: voyage-2, voyage-large-2, voyage-code-2
  
- **HuggingFace**:
  - `HUGGINGFACE_API_KEY`: Your HuggingFace API key
  - Any model from HuggingFace Hub
  
- **Cohere**:
  - `COHERE_API_KEY`: Your Cohere API key
  - Models: embed-english-v3.0, embed-multilingual-v3.0

**Chunking Configuration**:
- `--chunk-size`: Characters per chunk (0 = no chunking)
- `--chunk-overlap`: Overlap between chunks for continuity
- `--chunk-strategy`: sliding, sentence, or paragraph
- `--aggregation-method`: mean, weighted_mean, max_pool, first_last, pca
- `--role-weights`: JSON dict for weighting user vs assistant chunks
- `--position-decay`: Decay factor for position-based weighting

**Memory-Efficient TF-IDF**:
- `--max-features`: Limit vocabulary size (default 5000)
- `--batch-size`: Process documents in batches (default 100)
- Automatically adjusts min_df for small datasets

### Data Format

**Input Format** (for chat log integrations):
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Node Format** (after import):
```json
{
  "id": "unique_identifier",
  "type": "conversation|bookmark|playlist_item",
  "content": {...},
  "metadata": {...},
  "timestamp": "2024-01-01T12:00:00"
}
```

**Embedding Format** (separate file):
```json
{
  "source_file": "/path/to/original/node.json",
  "node_id": "unique_identifier",
  "node_type": "conversation",
  "embedding": [0.1, 0.2, ...],
  "embedding_method": "llm:openai",
  "embedding_model": "text-embedding-3-large",
  "timestamp": "2024-01-01T12:00:00",
  "metadata": {
    "provider": "openai",
    "chunks": 5,
    "aggregation": "weighted_mean"
  }
}
```

## Adding Custom Integrations

To add a new integration:

1. Create a new module in `integrations/your_type/`
2. Implement the `DataSource` interface
3. Register it in `integrations/your_type/__init__.py`
4. The integration will be automatically discovered

Example:
```python
from integrations.base import DataSource, Node, Edge

class YourIntegration(DataSource):
    def load(self, source: str):
        # Load your data
        pass
    
    def extract_nodes(self):
        # Yield Node objects
        pass
    
    def extract_edges(self, nodes=None):
        # Yield Edge objects (optional)
        pass
    
    def get_node_content_for_embedding(self, node):
        # Return text content for embedding
        pass
```