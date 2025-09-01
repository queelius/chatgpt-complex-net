# Proposed Redesign Without Backward Compatibility

## Current Issues
1. Mixed data storage patterns (JSON files, embeddings separate/embedded)
2. Manual multi-step process (import → embed → edges → export)
3. Redundant code paths (efficient vs standard TF-IDF)
4. No caching or incremental updates
5. No metadata tracking or versioning

## Proposed Architecture

### 1. Unified Data Store
Use SQLite database as single source of truth:

```sql
-- Nodes table
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    source_integration TEXT,
    content JSON,
    metadata JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Embeddings table (supports multiple embeddings per node)
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    node_id TEXT REFERENCES nodes(id),
    method TEXT,  -- 'tfidf', 'openai', 'ollama', etc.
    model TEXT,   -- specific model used
    vector BLOB,  -- compressed numpy array
    metadata JSON,
    created_at TIMESTAMP,
    UNIQUE(node_id, method, model)
);

-- Edges table (can store different edge types)
CREATE TABLE edges (
    source_id TEXT REFERENCES nodes(id),
    target_id TEXT REFERENCES nodes(id),
    edge_type TEXT,  -- 'similarity', 'temporal', 'reference', etc.
    weight REAL,
    metadata JSON,
    created_at TIMESTAMP,
    PRIMARY KEY(source_id, target_id, edge_type)
);

-- Processing history for incremental updates
CREATE TABLE processing_log (
    id INTEGER PRIMARY KEY,
    operation TEXT,
    parameters JSON,
    status TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### 2. Simplified CLI Commands

```bash
# Single command to process everything
semnet process <source> --integration chatlog --embedding openai --output graph.gexf

# Or with pipeline config
semnet pipeline config.yaml

# Incremental updates (only process new data)
semnet update <database.db> <new_source>

# Interactive exploration
semnet explore <database.db>
```

### 3. Pipeline Configuration (YAML)

```yaml
name: "conversation-analysis"
database: "conversations.db"

sources:
  - integration: chatlog.json
    path: ./data/conversations
    options:
      link_strategy: temporal
      
  - integration: chatlog.markdown
    path: ./data/markdown_logs

embeddings:
  - method: openai
    model: text-embedding-3-large
    cache: true
    
  - method: tfidf
    max_features: 5000
    
edges:
  - type: similarity
    threshold: 0.7
    embedding: openai
    
  - type: temporal
    window: 3600
    
exports:
  - format: gexf
    output: graph.gexf
    include_embeddings: false
    
  - format: json
    output: nodes.json
    
  - format: csv
    output: edges.csv
```

### 4. Powerful New Features

#### A. Incremental Processing
```python
class IncrementalProcessor:
    def process_new(self, source):
        # Only process files/data not seen before
        new_items = self.detect_new(source)
        if new_items:
            self.process(new_items)
```

#### B. Multi-Modal Embeddings
```python
class MultiModalEmbedding:
    def embed_node(self, node):
        # Generate multiple embeddings per node
        embeddings = {
            'content': self.text_embedding(node.content),
            'title': self.text_embedding(node.title),
            'semantic': self.llm_embedding(node.full_text),
            'structural': self.graph_embedding(node.connections)
        }
        return embeddings
```

#### C. Smart Edge Generation
```python
class SmartEdgeGenerator:
    def generate_edges(self, nodes):
        edges = []
        
        # Similarity edges
        edges.extend(self.similarity_edges(nodes, threshold=0.7))
        
        # Reference edges (one node mentions another)
        edges.extend(self.reference_edges(nodes))
        
        # Topical edges (shared topics/tags)
        edges.extend(self.topic_edges(nodes))
        
        # Temporal edges
        edges.extend(self.temporal_edges(nodes))
        
        return self.deduplicate_and_weight(edges)
```

#### D. Query Interface
```python
# Find similar nodes
similar = db.query("""
    SELECT * FROM nodes 
    WHERE similarity_to(:node_id) > 0.8
    ORDER BY similarity DESC
    LIMIT 10
""")

# Complex graph queries
connected = db.query("""
    WITH RECURSIVE connected AS (
        SELECT target_id, weight, 1 as depth
        FROM edges WHERE source_id = :start_node
        UNION ALL
        SELECT e.target_id, e.weight * c.weight, c.depth + 1
        FROM edges e JOIN connected c ON e.source_id = c.target_id
        WHERE c.depth < 3
    )
    SELECT * FROM connected ORDER BY weight DESC
""")
```

### 5. Advanced Features

#### A. Embedding Versioning
Track different embedding versions and compare results:
```bash
semnet compare-embeddings --method1 openai --method2 ollama
```

#### B. Active Learning
Identify nodes that need better embeddings:
```bash
semnet suggest-reprocess --confidence-threshold 0.5
```

#### C. Graph Analysis Suite
Built-in analysis without exporting:
```bash
semnet analyze communities --algorithm louvain
semnet analyze centrality --method pagerank
semnet analyze paths --from node1 --to node2
```

#### D. Real-time Updates
Watch directories for changes:
```bash
semnet watch ./data --auto-process
```

### 6. Benefits of This Approach

1. **Single source of truth** - SQLite database contains everything
2. **Incremental updates** - Only process what's new
3. **Multiple embeddings** - Can compare different methods
4. **Rich queries** - SQL + graph operations
5. **Caching** - Automatic deduplication
6. **Versioning** - Track changes over time
7. **Performance** - Indexed database vs file scanning
8. **Atomicity** - Transaction support
9. **Portability** - Single file to share/backup
10. **Extensibility** - Easy to add new tables/features

### 7. Migration Path

```bash
# Convert existing data
semnet migrate ./data/imported ./data/embeddings --output database.db

# Or import fresh
semnet init database.db
semnet import database.db --source ./data --integration chatlog
```

## Implementation Priority

1. **Phase 1**: Database schema and basic operations
2. **Phase 2**: Pipeline configuration and automation
3. **Phase 3**: Advanced queries and analysis
4. **Phase 4**: Real-time and incremental features
5. **Phase 5**: Web UI for exploration

This design eliminates backward compatibility constraints and creates a much more powerful, efficient system.