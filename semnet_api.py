"""
Semantic Network API - Core functionality exposed as a Pythonic API.

This module provides the core semantic network operations as a clean API
that can be used programmatically or via the CLI.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Types
# ============================================================================

@dataclass
class Node:
    """Represents a node in the semantic network."""
    id: str
    type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'content': self.content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        return cls(
            id=data['id'],
            type=data['type'],
            content=data.get('content', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class Embedding:
    """Represents an embedding for a node."""
    node_id: str
    vector: List[float]
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'embedding': self.vector,
            'embedding_method': self.method,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Embedding':
        return cls(
            node_id=data['node_id'],
            vector=data['embedding'],
            method=data.get('embedding_method', 'unknown'),
            metadata=data.get('metadata', {})
        )


@dataclass
class Edge:
    """Represents an edge in the semantic network."""
    source: str
    target: str
    weight: float
    type: str = 'similarity'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'type': self.type,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        return cls(
            source=data['source'],
            target=data['target'],
            weight=data['weight'],
            type=data.get('type', 'similarity'),
            metadata=data.get('metadata', {})
        )


class EmbeddingMethod(Enum):
    """Available embedding methods."""
    TFIDF = 'tfidf'
    LLM = 'llm'


# ============================================================================
# Core API Functions
# ============================================================================

class DataImporter:
    """Handles importing data from various sources."""
    
    def __init__(self, integration_name: str, config: Optional[Dict] = None):
        """Initialize importer with a specific integration."""
        from integrations import load_integration
        self.integration = load_integration(integration_name, config=config)
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
    
    def load(self, source: str) -> 'DataImporter':
        """Load data from source. Returns self for chaining."""
        logger.info(f"Loading data from {source} using {self.integration.__class__.__name__}")
        self.integration.load(source)
        return self
    
    def extract_nodes(self) -> List[Node]:
        """Extract nodes from loaded data."""
        self.nodes = [
            Node(
                id=n.id,
                type=n.type,
                content=n.content,
                metadata=n.metadata
            )
            for n in self.integration.extract_nodes()
        ]
        logger.info(f"Extracted {len(self.nodes)} nodes")
        return self.nodes
    
    def extract_edges(self) -> List[Edge]:
        """Extract edges from loaded data."""
        integration_edges = self.integration.extract_edges(
            [n for n in self.integration.extract_nodes()]
        )
        self.edges = [
            Edge(
                source=e.source,
                target=e.target,
                weight=e.weight,
                type=e.type,
                metadata=getattr(e, 'metadata', {})
            )
            for e in integration_edges
        ]
        logger.info(f"Extracted {len(self.edges)} edges")
        return self.edges
    
    def get_node_content_for_embedding(self, node: Node) -> str:
        """Get text content for embedding from a node."""
        # Convert back to integration node format temporarily
        integration_node = type('Node', (), {
            'id': node.id,
            'type': node.type,
            'content': node.content,
            'metadata': node.metadata,
            'to_dict': lambda: node.to_dict()
        })()
        return self.integration.get_node_content_for_embedding(integration_node)


class EmbeddingGenerator:
    """Handles embedding generation for nodes."""
    
    def __init__(self, method: EmbeddingMethod = EmbeddingMethod.TFIDF):
        """Initialize with embedding method."""
        self.method = method
        self.embeddings: List[Embedding] = []
    
    def generate_tfidf_embeddings(
        self,
        nodes: List[Node],
        content_extractor: callable,
        max_features: int = 5000,
        batch_size: int = 100
    ) -> List[Embedding]:
        """Generate TF-IDF embeddings for nodes."""
        from embedding.tfidf_memory_efficient import MemoryEfficientTfidfVectorizer
        
        logger.info(f"Generating TF-IDF embeddings for {len(nodes)} nodes")
        
        # Extract text content
        texts = [content_extractor(node) for node in nodes]
        
        # Generate embeddings
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=max_features)
        vectorizer.fit(texts)
        
        self.embeddings = []
        for idx, vector in vectorizer.transform_batch(texts, batch_size=batch_size):
            self.embeddings.append(Embedding(
                node_id=nodes[idx].id,
                vector=vector,
                method='tfidf',
                metadata={'max_features': max_features}
            ))
        
        return self.embeddings
    
    def generate_llm_embeddings(
        self,
        nodes: List[Node],
        content_extractor: callable,
        provider: str = 'ollama',
        config: Optional[Dict] = None,
        chunk_size: int = 0,
        chunk_overlap: int = 50
    ) -> List[Embedding]:
        """Generate LLM embeddings for nodes."""
        from embedding.llm_providers import get_embedding_provider
        from embedding.chunking import ChunkedEmbeddingProcessor, ChunkConfig, AggregationConfig
        
        logger.info(f"Generating LLM embeddings for {len(nodes)} nodes using {provider}")
        
        # Get provider
        provider_instance = get_embedding_provider(provider, config or {})
        
        self.embeddings = []
        
        if chunk_size > 0:
            # Use chunking
            chunk_config = ChunkConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy='sliding'
            )
            agg_config = AggregationConfig(method='mean')
            processor = ChunkedEmbeddingProcessor(provider_instance, chunk_config, agg_config)
            
            for node in nodes:
                text = content_extractor(node)
                vector, metadata = processor.process_text(text)
                self.embeddings.append(Embedding(
                    node_id=node.id,
                    vector=vector,
                    method=f'llm:{provider}',
                    metadata=metadata
                ))
        else:
            # Direct embedding
            for node in nodes:
                text = content_extractor(node)
                vector = provider_instance.get_embedding(text)
                self.embeddings.append(Embedding(
                    node_id=node.id,
                    vector=vector,
                    method=f'llm:{provider}',
                    metadata={'provider': provider}
                ))
        
        return self.embeddings


class EdgeGenerator:
    """Handles edge generation from embeddings."""
    
    @staticmethod
    def compute_similarity_edges(
        embeddings: List[Embedding],
        threshold: float = 0.7
    ) -> List[Edge]:
        """Compute edges based on cosine similarity between embeddings."""
        if not embeddings:
            return []
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info(f"Computing similarity edges for {len(embeddings)} embeddings")
        
        # Extract vectors and normalize
        vectors = np.array([e.vector for e in embeddings])
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(vectors_norm)
        
        # Generate edges above threshold
        edges = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    edges.append(Edge(
                        source=embeddings[i].node_id,
                        target=embeddings[j].node_id,
                        weight=float(sim),
                        type='similarity'
                    ))
        
        logger.info(f"Generated {len(edges)} edges (threshold={threshold})")
        return edges


class GraphExporter:
    """Handles exporting the semantic network to various formats."""
    
    @staticmethod
    def export_to_gexf(
        nodes: List[Node],
        edges: List[Edge],
        output_file: Path
    ) -> None:
        """Export graph to GEXF format."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node.id, **node.metadata, type=node.type)
        
        # Add edges
        for edge in edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        nx.write_gexf(G, str(output_file))
        logger.info(f"Exported graph to {output_file}")
    
    @staticmethod
    def export_to_json(
        nodes: List[Node],
        edges: List[Edge],
        output_file: Path
    ) -> None:
        """Export graph to JSON format."""
        graph_data = {
            'nodes': [n.to_dict() for n in nodes],
            'edges': [e.to_dict() for e in edges]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported graph to {output_file}")


# ============================================================================
# Storage Functions
# ============================================================================

class Storage:
    """Handles saving and loading data to/from disk."""
    
    @staticmethod
    def save_nodes(nodes: List[Node], output_dir: Path) -> None:
        """Save nodes to individual JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for node in nodes:
            node_file = output_dir / f"{node.id}.json"
            with open(node_file, 'w', encoding='utf-8') as f:
                json.dump(node.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Saved {len(nodes)} nodes to {output_dir}")
    
    @staticmethod
    def load_nodes(input_dir: Path) -> List[Node]:
        """Load nodes from JSON files."""
        node_files = [f for f in input_dir.glob("*.json") 
                     if f.name not in ['edges.json', 'import_stats.json']]
        
        nodes = []
        for node_file in node_files:
            with open(node_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                nodes.append(Node.from_dict(data))
        
        logger.info(f"Loaded {len(nodes)} nodes from {input_dir}")
        return nodes
    
    @staticmethod
    def save_embeddings(embeddings: List[Embedding], output_dir: Path) -> None:
        """Save embeddings to individual JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for embedding in embeddings:
            embed_file = output_dir / f"{embedding.node_id}.json"
            with open(embed_file, 'w', encoding='utf-8') as f:
                json.dump(embedding.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {output_dir}")
    
    @staticmethod
    def load_embeddings(input_dir: Path) -> List[Embedding]:
        """Load embeddings from JSON files."""
        embedding_files = [f for f in input_dir.glob("*.json")
                          if f.name not in ['edges.json', 'import_stats.json']]
        
        embeddings = []
        for embed_file in embedding_files:
            with open(embed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'embedding' in data and 'node_id' in data:
                    embeddings.append(Embedding.from_dict(data))
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {input_dir}")
        return embeddings
    
    @staticmethod
    def save_edges(edges: List[Edge], output_file: Path) -> None:
        """Save edges to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([e.to_dict() for e in edges], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(edges)} edges to {output_file}")
    
    @staticmethod
    def load_edges(input_file: Path) -> List[Edge]:
        """Load edges from a JSON file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        edges = [Edge.from_dict(e) for e in data]
        logger.info(f"Loaded {len(edges)} edges from {input_file}")
        return edges


# ============================================================================
# Fluent API Interface
# ============================================================================

class SemanticNetwork:
    """
    Fluent API for building semantic networks.
    
    Example:
        network = SemanticNetwork()
        network.import_data('chatlog.json', '/path/to/logs') \
               .generate_embeddings(method='llm', provider='openai') \
               .compute_edges(threshold=0.8) \
               .export('output.gexf')
    """
    
    def __init__(self):
        """Initialize an empty semantic network."""
        self.nodes: List[Node] = []
        self.embeddings: List[Embedding] = []
        self.edges: List[Edge] = []
        self.importer: Optional[DataImporter] = None
    
    def import_data(
        self,
        integration: str,
        source: str,
        config: Optional[Dict] = None,
        extract_edges: bool = False
    ) -> 'SemanticNetwork':
        """Import data using specified integration."""
        self.importer = DataImporter(integration, config)
        self.importer.load(source)
        self.nodes = self.importer.extract_nodes()
        
        if extract_edges:
            self.edges.extend(self.importer.extract_edges())
        
        return self
    
    def load_nodes(self, input_dir: Union[str, Path]) -> 'SemanticNetwork':
        """Load nodes from directory."""
        self.nodes = Storage.load_nodes(Path(input_dir))
        return self
    
    def save_nodes(self, output_dir: Union[str, Path]) -> 'SemanticNetwork':
        """Save nodes to directory."""
        Storage.save_nodes(self.nodes, Path(output_dir))
        return self
    
    def generate_embeddings(
        self,
        method: str = 'tfidf',
        **kwargs
    ) -> 'SemanticNetwork':
        """Generate embeddings for nodes."""
        if not self.nodes:
            raise ValueError("No nodes loaded. Call import_data() or load_nodes() first.")
        
        if not self.importer:
            raise ValueError("No importer available. Use import_data() to load data first.")
        
        generator = EmbeddingGenerator(EmbeddingMethod(method))
        
        if method == 'tfidf':
            self.embeddings = generator.generate_tfidf_embeddings(
                self.nodes,
                self.importer.get_node_content_for_embedding,
                **kwargs
            )
        elif method == 'llm':
            self.embeddings = generator.generate_llm_embeddings(
                self.nodes,
                self.importer.get_node_content_for_embedding,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        return self
    
    def load_embeddings(self, input_dir: Union[str, Path]) -> 'SemanticNetwork':
        """Load embeddings from directory."""
        self.embeddings = Storage.load_embeddings(Path(input_dir))
        return self
    
    def save_embeddings(self, output_dir: Union[str, Path]) -> 'SemanticNetwork':
        """Save embeddings to directory."""
        Storage.save_embeddings(self.embeddings, Path(output_dir))
        return self
    
    def compute_edges(self, threshold: float = 0.7) -> 'SemanticNetwork':
        """Compute edges from embeddings."""
        if not self.embeddings:
            raise ValueError("No embeddings available. Generate or load embeddings first.")
        
        similarity_edges = EdgeGenerator.compute_similarity_edges(self.embeddings, threshold)
        self.edges.extend(similarity_edges)
        return self
    
    def load_edges(self, input_file: Union[str, Path]) -> 'SemanticNetwork':
        """Load edges from file."""
        self.edges = Storage.load_edges(Path(input_file))
        return self
    
    def save_edges(self, output_file: Union[str, Path]) -> 'SemanticNetwork':
        """Save edges to file."""
        Storage.save_edges(self.edges, Path(output_file))
        return self
    
    def export(
        self,
        output_file: Union[str, Path],
        format: str = 'gexf'
    ) -> 'SemanticNetwork':
        """Export the network to specified format."""
        output_path = Path(output_file)
        
        if format == 'gexf':
            GraphExporter.export_to_gexf(self.nodes, self.edges, output_path)
        elif format == 'json':
            GraphExporter.export_to_json(self.nodes, self.edges, output_path)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        return self
    
    def pipeline(
        self,
        integration: str,
        source: str,
        output_file: Union[str, Path],
        embedding_method: str = 'tfidf',
        threshold: float = 0.7,
        **kwargs
    ) -> 'SemanticNetwork':
        """
        Run complete pipeline in one call.
        
        Example:
            network = SemanticNetwork()
            network.pipeline(
                'chatlog.json',
                '/path/to/logs',
                'output.gexf',
                embedding_method='llm',
                provider='openai',
                threshold=0.8
            )
        """
        self.import_data(integration, source, extract_edges=True)
        self.generate_embeddings(method=embedding_method, **kwargs)
        self.compute_edges(threshold=threshold)
        self.export(output_file)
        return self


# ============================================================================
# Convenience Functions (for direct use)
# ============================================================================

def import_data(integration: str, source: str, output_dir: Union[str, Path], **kwargs) -> List[Node]:
    """Import data and save nodes to directory."""
    importer = DataImporter(integration, kwargs.get('config'))
    importer.load(source)
    nodes = importer.extract_nodes()
    Storage.save_nodes(nodes, Path(output_dir))
    return nodes


def generate_embeddings(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    integration: str,
    method: str = 'tfidf',
    **kwargs
) -> List[Embedding]:
    """Generate embeddings for nodes in directory."""
    nodes = Storage.load_nodes(Path(input_dir))
    importer = DataImporter(integration)
    
    generator = EmbeddingGenerator(EmbeddingMethod(method))
    
    if method == 'tfidf':
        embeddings = generator.generate_tfidf_embeddings(
            nodes,
            lambda n: importer.get_node_content_for_embedding(n),
            **kwargs
        )
    else:
        embeddings = generator.generate_llm_embeddings(
            nodes,
            lambda n: importer.get_node_content_for_embedding(n),
            **kwargs
        )
    
    Storage.save_embeddings(embeddings, Path(output_dir))
    return embeddings


def compute_edges(
    input_dir: Union[str, Path],
    output_file: Union[str, Path],
    threshold: float = 0.7
) -> List[Edge]:
    """Compute edges from embeddings in directory."""
    embeddings = Storage.load_embeddings(Path(input_dir))
    edges = EdgeGenerator.compute_similarity_edges(embeddings, threshold)
    Storage.save_edges(edges, Path(output_file))
    return edges


def export_graph(
    nodes_dir: Union[str, Path],
    edges_file: Union[str, Path],
    output_file: Union[str, Path],
    format: str = 'gexf'
) -> None:
    """Export graph from nodes and edges files."""
    nodes = Storage.load_nodes(Path(nodes_dir))
    edges = Storage.load_edges(Path(edges_file))
    
    if format == 'gexf':
        GraphExporter.export_to_gexf(nodes, edges, Path(output_file))
    elif format == 'json':
        GraphExporter.export_to_json(nodes, edges, Path(output_file))
    else:
        raise ValueError(f"Unknown format: {format}")