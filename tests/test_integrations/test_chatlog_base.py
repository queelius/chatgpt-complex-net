"""
Tests for chatlog base integration.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from integrations.chatlog.base import ChatLogSource
from integrations.base import Node, Edge


class MockChatLogSource(ChatLogSource):
    """Concrete implementation for testing."""
    
    def load(self, source):
        """Mock load implementation."""
        pass
    
    def extract_nodes(self):
        """Mock extract_nodes implementation."""
        return iter([])


class TestChatLogSource:
    """Test ChatLogSource base class."""
    
    @pytest.fixture
    def chatlog_source(self):
        """Create a ChatLogSource instance."""
        return MockChatLogSource()
    
    @pytest.fixture
    def sample_nodes(self):
        """Create sample conversation nodes."""
        base_time = datetime(2024, 1, 1, 12, 0)
        nodes = []
        
        for i in range(3):
            node = Node(
                id=f"conv_{i}",
                type="conversation",
                content={
                    "messages": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Answer {i}"}
                    ]
                },
                metadata={"topic": f"topic_{i % 2}"},  # Two different topics
                timestamp=base_time + timedelta(minutes=i * 30)
            )
            nodes.append(node)
        
        return nodes
    
    @pytest.mark.unit
    def test_initialization_default(self):
        """Test default initialization."""
        source = MockChatLogSource()
        
        assert source.conversations == []
        assert source.link_strategy == 'temporal'
        assert source.temporal_window == 3600
    
    @pytest.mark.unit
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {
            'link_strategy': 'sequential',
            'temporal_window': 7200,
            'role_weights': {'user': 2.0, 'assistant': 1.0}
        }
        source = MockChatLogSource(config)
        
        assert source.link_strategy == 'sequential'
        assert source.temporal_window == 7200
        assert source.config == config
    
    @pytest.mark.unit
    def test_get_node_content_for_embedding_basic(self, chatlog_source):
        """Test extracting content for embedding from a node."""
        node = Node(
            id="test",
            type="conversation",
            content={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}
                ]
            }
        )
        
        content = chatlog_source.get_node_content_for_embedding(node)
        
        # Default weights: user=1.5, assistant=1.0
        # User content should appear with weight
        assert "Hello" in content
        assert "Hi there" in content
    
    @pytest.mark.unit
    def test_get_node_content_with_role_weights(self):
        """Test content extraction with custom role weights."""
        config = {'role_weights': {'user': 2.0, 'assistant': 1.0, 'system': 0.5}}
        source = MockChatLogSource(config)
        
        node = Node(
            id="test",
            type="conversation",
            content={
                "messages": [
                    {"role": "user", "content": "User message"},
                    {"role": "assistant", "content": "Assistant message"},
                    {"role": "system", "content": "System message"}
                ]
            }
        )
        
        content = source.get_node_content_for_embedding(node)
        
        # User message should appear twice (weight=2.0)
        assert content.count("User message") == 2
        assert content.count("Assistant message") == 1
        assert content.count("System message") == 1
    
    @pytest.mark.unit
    def test_get_node_content_empty_messages(self, chatlog_source):
        """Test handling node with no messages."""
        node = Node(
            id="empty",
            type="conversation",
            content={"messages": []}
        )
        
        content = chatlog_source.get_node_content_for_embedding(node)
        assert content == ""
    
    @pytest.mark.unit
    def test_get_node_content_missing_fields(self, chatlog_source):
        """Test handling messages with missing fields."""
        node = Node(
            id="test",
            type="conversation",
            content={
                "messages": [
                    {"content": "No role field"},  # Missing role
                    {"role": "user"},  # Missing content
                    {"role": "assistant", "content": "Normal message"}
                ]
            }
        )
        
        content = chatlog_source.get_node_content_for_embedding(node)
        
        # Should handle missing fields gracefully
        assert "No role field" in content
        assert "Normal message" in content
    
    @pytest.mark.unit
    def test_extract_edges_temporal_strategy(self, sample_nodes):
        """Test temporal edge creation strategy."""
        source = MockChatLogSource({'link_strategy': 'temporal', 'temporal_window': 3600})
        
        edges = source.extract_edges(sample_nodes)
        
        # Nodes are 30 minutes apart, within 3600 second (1 hour) window
        # Should create edges between adjacent conversations
        assert len(edges) > 0
        
        # Check edge structure
        for edge in edges:
            assert isinstance(edge, Edge)
            assert edge.type == 'temporal'
    
    @pytest.mark.unit
    def test_extract_edges_sequential_strategy(self, sample_nodes):
        """Test sequential edge creation strategy."""
        source = MockChatLogSource({'link_strategy': 'sequential'})
        
        edges = source.extract_edges(sample_nodes)
        
        # Sequential should link conversations in order
        assert len(edges) == len(sample_nodes) - 1  # n-1 edges for n nodes
        
        for edge in edges:
            assert edge.type == 'sequential'
    
    @pytest.mark.unit
    def test_extract_edges_none_strategy(self, sample_nodes):
        """Test no edge creation strategy."""
        source = MockChatLogSource({'link_strategy': 'none'})
        
        edges = source.extract_edges(sample_nodes)
        
        assert edges == []
    
    @pytest.mark.unit
    def test_temporal_edges_with_missing_timestamps(self):
        """Test temporal edges when some nodes lack timestamps."""
        source = MockChatLogSource({'link_strategy': 'temporal'})
        
        nodes = [
            Node(id="1", type="conv", content={}, timestamp=datetime(2024, 1, 1, 12, 0)),
            Node(id="2", type="conv", content={}, timestamp=None),  # No timestamp
            Node(id="3", type="conv", content={}, timestamp=datetime(2024, 1, 1, 12, 30))
        ]
        
        edges = source.extract_edges(nodes)
        
        # Should skip nodes without timestamps
        edge_ids = [(e.source_id, e.target_id) for e in edges]
        assert ("2", "1") not in edge_ids
        assert ("2", "3") not in edge_ids
    
    @pytest.mark.unit
    def test_temporal_edges_outside_window(self):
        """Test temporal edges with conversations outside time window."""
        source = MockChatLogSource({'link_strategy': 'temporal', 'temporal_window': 1800})  # 30 minutes
        
        base_time = datetime(2024, 1, 1, 12, 0)
        nodes = [
            Node(id="1", type="conv", content={}, timestamp=base_time),
            Node(id="2", type="conv", content={}, timestamp=base_time + timedelta(minutes=20)),  # Within window
            Node(id="3", type="conv", content={}, timestamp=base_time + timedelta(hours=2))  # Outside window
        ]
        
        edges = source.extract_edges(nodes)
        
        # Should only create edge between nodes 1 and 2
        edge_pairs = {(e.source_id, e.target_id) for e in edges}
        assert ("1", "2") in edge_pairs or ("2", "1") in edge_pairs
        assert ("1", "3") not in edge_pairs
        assert ("3", "1") not in edge_pairs
    
    @pytest.mark.unit
    def test_sequential_edges_ordering(self):
        """Test that sequential edges maintain proper order."""
        source = MockChatLogSource({'link_strategy': 'sequential'})
        
        base_time = datetime(2024, 1, 1, 12, 0)
        nodes = [
            Node(id="3", type="conv", content={}, timestamp=base_time + timedelta(hours=2)),
            Node(id="1", type="conv", content={}, timestamp=base_time),
            Node(id="2", type="conv", content={}, timestamp=base_time + timedelta(hours=1))
        ]
        
        edges = source.extract_edges(nodes)
        
        # Should create edges in timestamp order: 1->2, 2->3
        edge_pairs = [(e.source_id, e.target_id) for e in edges]
        assert ("1", "2") in edge_pairs
        assert ("2", "3") in edge_pairs
        assert len(edges) == 2