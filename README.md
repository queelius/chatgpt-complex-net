# ChatGPT Complex Networks

A toolkit for generating, analyzing, and visualizing complex networks from conversational data. This project provides tools for embedding conversations, generating similarity-based edges, and performing network analysis on conversation graphs.

## Features

- **Flexible Embedding Generation**:
  - LLM embedding support via Ollama API
  - TF-IDF embeddings for classical NLP approach
  - Role-based aggregation (separate embeddings for users vs assistants)
  - Automatic chunking for long messages

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
git clone https://github.com/queelius/chatgpt-complex-net.git
cd chatgpt-complex-net

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generating Node Embeddings

```bash
# Generate role-based embeddings using LLM model
python cli.py node-embeddings --input-dir ./data/conversations \
    --method role-aggregate --embedding-method llm
    
# Generate chunked embeddings for large documents
python cli.py node-embeddings --input-dir ./data/conversations \
    --method chunked --chunk-size 512
```

### Creating Graph Edges

```bash
# Generate edges (CPU version)
python cli.py edges --input-dir ./data/embeddings_json \
    --output-file edges.json

# GPU-accelerated edge generation
python cli.py edges-gpu --input-dir ./data/embeddings_json \
    --output-file edges.json

# Filter edges by similarity threshold
python cli.py cut-off --input-file edges.json \
    --output-file filtered_edges.json --cutoff 0.7
```

### Exporting and Visualization

```bash
# Export to Gephi-compatible format
python cli.py export --nodes-dir ./data/embeddings_json \
    --edges-file filtered_edges.json --format gexf \
    --output-file graph.gexf
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

- cli.py - Command-line interface for all operations
- networks.py - Core network generation functionality
- rec-conv.py - Recommendation system
- graph - Graph processing utilities
  - edge_utils.py - Edge generation and filtering
  - export_utils.py - Graph format conversion
  - normalization.py - Vector normalization utilities
  - gpu-edge.py - GPU-accelerated edge generation
- embedding - Embedding generation
  - llm_embedding_model.py - LLM-based embeddings
  - tdidf_embedding_model.py - Classical TF-IDF embeddings
  - utils.py - Text chunking and processing
- data - Data processing
  - message_utils.py - Message formatting utilities

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