"""
Base classes and interfaces for data source integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterator, Tuple
from datetime import datetime
import json

@dataclass
class Node:
    """Represents a node in the semantic network."""
    id: str
    type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "embedding": self.embedding
        }

@dataclass
class Edge:
    """Represents an edge/link between nodes."""
    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert edge to dictionary for serialization."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "metadata": self.metadata
        }

class DataSource(ABC):
    """
    Abstract base class for all data source integrations.
    Each integration must implement methods to extract nodes and edges.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data source with optional configuration.
        
        Args:
            config: Integration-specific configuration parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def load(self, source: Any) -> None:
        """
        Load data from the source (file path, URL, API endpoint, etc.)
        
        Args:
            source: The data source identifier (type depends on integration)
        """
        pass
    
    @abstractmethod
    def extract_nodes(self) -> Iterator[Node]:
        """
        Extract nodes from the loaded data.
        
        Yields:
            Node objects representing entities in the data
        """
        pass
    
    @abstractmethod
    def extract_edges(self, nodes: Optional[List[Node]] = None) -> Iterator[Edge]:
        """
        Extract edges/relationships from the data.
        
        Args:
            nodes: Optional list of nodes to constrain edge extraction
            
        Yields:
            Edge objects representing relationships
        """
        pass
    
    @abstractmethod
    def get_node_content_for_embedding(self, node: Node) -> str:
        """
        Get the text content of a node that should be used for embedding generation.
        
        Args:
            node: The node to extract content from
            
        Returns:
            Text string to be embedded
        """
        pass
    
    def transform_for_export(self, node: Node) -> Dict[str, Any]:
        """
        Transform a node for export to the existing graph format.
        Override this to customize how nodes are exported.
        
        Args:
            node: The node to transform
            
        Returns:
            Dictionary in the expected export format
        """
        return {
            "id": node.id,
            "messages": node.content.get("messages", []),
            "metadata": node.metadata,
            "timestamp": node.timestamp.isoformat() if node.timestamp else None
        }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the loaded data and configuration.
        
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        return True, []
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary of statistics (node count, edge count, etc.)
        """
        return {
            "integration": self.name,
            "config": self.config
        }

class IntegrationRegistry:
    """Registry for managing available data source integrations."""
    
    _integrations: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, integration_class: type) -> None:
        """
        Register a new integration.
        
        Args:
            name: Unique name for the integration
            integration_class: The DataSource subclass
        """
        if not issubclass(integration_class, DataSource):
            raise TypeError(f"{integration_class} must be a subclass of DataSource")
        cls._integrations[name] = integration_class
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get an integration by name."""
        return cls._integrations.get(name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered integration names."""
        return list(cls._integrations.keys())
    
    @classmethod
    def create_instance(cls, name: str, config: Optional[Dict[str, Any]] = None) -> DataSource:
        """
        Create an instance of an integration.
        
        Args:
            name: Name of the integration
            config: Configuration for the integration
            
        Returns:
            Initialized DataSource instance
        """
        integration_class = cls.get(name)
        if not integration_class:
            raise ValueError(f"Integration '{name}' not found")
        return integration_class(config)