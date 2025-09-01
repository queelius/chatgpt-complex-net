"""
Tests for Node and Edge data structures.
"""

import pytest
import json
from datetime import datetime

from integrations.base import Node, Edge


class TestNode:
    """Test Node data structure."""
    
    @pytest.mark.unit
    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(
            id="test_001",
            type="document",
            content={"text": "Sample text"},
            metadata={"author": "test"}
        )
        
        assert node.id == "test_001"
        assert node.type == "document"
        assert node.content["text"] == "Sample text"
        assert node.metadata["author"] == "test"
        assert node.timestamp is None
        assert node.embedding is None
    
    @pytest.mark.unit
    def test_node_with_timestamp(self):
        """Test node with timestamp."""
        now = datetime.now()
        node = Node(
            id="test_002",
            type="conversation",
            content={},
            timestamp=now
        )
        
        assert node.timestamp == now
    
    @pytest.mark.unit
    def test_node_to_dict(self):
        """Test node serialization to dictionary."""
        now = datetime.now()
        node = Node(
            id="test_003",
            type="message",
            content={"text": "Hello"},
            metadata={"index": 1},
            timestamp=now,
            embedding=[0.1, 0.2, 0.3]
        )
        
        node_dict = node.to_dict()
        
        assert node_dict["id"] == "test_003"
        assert node_dict["type"] == "message"
        assert node_dict["content"]["text"] == "Hello"
        assert node_dict["metadata"]["index"] == 1
        assert node_dict["timestamp"] == now.isoformat()
        assert node_dict["embedding"] == [0.1, 0.2, 0.3]
    
    @pytest.mark.unit
    def test_node_to_dict_no_timestamp(self):
        """Test node serialization without timestamp."""
        node = Node(
            id="test_004",
            type="document",
            content={}
        )
        
        node_dict = node.to_dict()
        assert node_dict["timestamp"] is None
    
    @pytest.mark.unit
    def test_node_json_serializable(self):
        """Test that node dict is JSON serializable."""
        node = Node(
            id="test_005",
            type="document",
            content={"nested": {"data": "value"}},
            metadata={"list": [1, 2, 3]},
            timestamp=datetime.now()
        )
        
        node_dict = node.to_dict()
        json_str = json.dumps(node_dict)
        loaded = json.loads(json_str)
        
        assert loaded["id"] == "test_005"
        assert loaded["content"]["nested"]["data"] == "value"
        assert loaded["metadata"]["list"] == [1, 2, 3]


class TestEdge:
    """Test Edge data structure."""
    
    @pytest.mark.unit
    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge(
            source_id="node_001",
            target_id="node_002",
            type="similarity"
        )
        
        assert edge.source_id == "node_001"
        assert edge.target_id == "node_002"
        assert edge.type == "similarity"
        assert edge.weight == 1.0  # Default weight
        assert edge.metadata == {}  # Default empty metadata
    
    @pytest.mark.unit
    def test_edge_with_weight_and_metadata(self):
        """Test edge with custom weight and metadata."""
        edge = Edge(
            source_id="node_003",
            target_id="node_004",
            type="reference",
            weight=0.75,
            metadata={"computed_at": "2024-01-01", "method": "cosine"}
        )
        
        assert edge.weight == 0.75
        assert edge.metadata["computed_at"] == "2024-01-01"
        assert edge.metadata["method"] == "cosine"
    
    @pytest.mark.unit
    def test_edge_to_dict(self):
        """Test edge serialization to dictionary."""
        edge = Edge(
            source_id="node_005",
            target_id="node_006",
            type="temporal",
            weight=0.9,
            metadata={"order": 1}
        )
        
        edge_dict = edge.to_dict()
        
        assert edge_dict["source"] == "node_005"  # Note: key is "source" not "source_id"
        assert edge_dict["target"] == "node_006"  # Note: key is "target" not "target_id"
        assert edge_dict["type"] == "temporal"
        assert edge_dict["weight"] == 0.9
        assert edge_dict["metadata"]["order"] == 1
    
    @pytest.mark.unit
    def test_edge_json_serializable(self):
        """Test that edge dict is JSON serializable."""
        edge = Edge(
            source_id="node_007",
            target_id="node_008",
            type="similarity",
            weight=0.5,
            metadata={"tags": ["important", "verified"]}
        )
        
        edge_dict = edge.to_dict()
        json_str = json.dumps(edge_dict)
        loaded = json.loads(json_str)
        
        assert loaded["source"] == "node_007"
        assert loaded["target"] == "node_008"
        assert loaded["weight"] == 0.5
        assert loaded["metadata"]["tags"] == ["important", "verified"]
    
    @pytest.mark.unit
    def test_edge_equality(self):
        """Test edge comparison."""
        edge1 = Edge("a", "b", "similarity", 0.5)
        edge2 = Edge("a", "b", "similarity", 0.5)
        edge3 = Edge("a", "c", "similarity", 0.5)
        
        # Note: dataclasses provide equality by default
        assert edge1 == edge2
        assert edge1 != edge3