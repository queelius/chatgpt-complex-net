"""
Shared pytest fixtures for all tests.
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from faker import Faker

from integrations.base import Node, Edge
from embedding.llm_providers import EmbeddingProvider
from embedding.chunking import ChunkConfig, AggregationConfig

fake = Faker()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_node():
    """Create a sample node."""
    return Node(
        id="test_node_001",
        type="conversation",
        content={
            "title": "Test Conversation",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
        },
        metadata={"source": "test", "timestamp": "2024-01-01T00:00:00Z"}
    )


@pytest.fixture
def sample_nodes():
    """Create multiple sample nodes."""
    nodes = []
    for i in range(5):
        nodes.append(Node(
            id=f"node_{i:03d}",
            type="conversation",
            content={
                "title": f"Conversation {i}",
                "messages": [
                    {"role": "user", "content": fake.sentence()},
                    {"role": "assistant", "content": fake.paragraph()}
                ]
            },
            metadata={"index": i}
        ))
    return nodes


@pytest.fixture
def sample_edge():
    """Create a sample edge."""
    return Edge(
        source_id="node_001",
        target_id="node_002",
        type="similarity",
        weight=0.85,
        metadata={"computed_at": "2024-01-01T00:00:00Z"}
    )


@pytest.fixture
def sample_conversation_json(temp_dir):
    """Create a sample conversation JSON file."""
    conversation = {
        "messages": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
            {"role": "user", "content": "Can you give me an example?"},
            {"role": "assistant", "content": "Sure! Image recognition is a common example..."}
        ],
        "metadata": {
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "test"
        }
    }
    
    file_path = temp_dir / "conversation.json"
    with open(file_path, 'w') as f:
        json.dump(conversation, f)
    
    return file_path


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    np.random.seed(42)
    return np.random.randn(768).tolist()  # Standard embedding size


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = Mock(spec=EmbeddingProvider)
    provider.embedding_dimension = 768
    provider.max_text_length = 8192
    
    # Mock embedding generation
    def mock_get_embedding(text):
        np.random.seed(hash(text) % 1000)
        return np.random.randn(768)
    
    provider.get_embedding = Mock(side_effect=mock_get_embedding)
    provider.get_embeddings_batch = Mock(side_effect=lambda texts: [mock_get_embedding(t) for t in texts])
    
    return provider


@pytest.fixture
def mock_ollama_response():
    """Mock response from Ollama API."""
    return {
        "embedding": np.random.randn(768).tolist()
    }


@pytest.fixture
def chunk_config():
    """Create a default chunk configuration."""
    return ChunkConfig(
        chunk_size=512,
        overlap=50,
        strategy="sliding",
        respect_boundaries=True
    )


@pytest.fixture
def aggregation_config():
    """Create a default aggregation configuration."""
    return AggregationConfig(
        method="weighted_mean",
        role_weights={"user": 1.5, "assistant": 1.0},
        position_decay=0.1
    )


@pytest.fixture
def sample_embeddings_batch():
    """Create a batch of sample embeddings."""
    np.random.seed(42)
    return [np.random.randn(768) for _ in range(10)]


@pytest.fixture
def conversation_directory(temp_dir):
    """Create a directory with multiple conversation files."""
    for i in range(3):
        conversation = {
            "id": f"conv_{i:03d}",
            "messages": [
                {"role": "user", "content": fake.sentence()},
                {"role": "assistant", "content": fake.paragraph()},
            ] * 2,
            "metadata": {"index": i}
        }
        
        file_path = temp_dir / f"conversation_{i:03d}.json"
        with open(file_path, 'w') as f:
            json.dump(conversation, f)
    
    return temp_dir


@pytest.fixture
def mock_requests(monkeypatch):
    """Mock requests library for API calls."""
    mock_post = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {"embedding": np.random.randn(768).tolist()}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    
    monkeypatch.setattr("requests.post", mock_post)
    return mock_post


@pytest.fixture
def sample_tfidf_documents():
    """Create sample documents for TF-IDF testing."""
    return [
        "Machine learning is a method of data analysis",
        "Deep learning is a subset of machine learning",
        "Natural language processing uses machine learning",
        "Computer vision is another application of deep learning",
        "Data science encompasses machine learning and statistics"
    ]


@pytest.fixture
def integration_config():
    """Create a sample integration configuration."""
    return {
        "link_strategy": "temporal",
        "extract_metadata": True,
        "max_nodes": 100
    }


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_ollama: Tests requiring Ollama")
    config.addinivalue_line("markers", "requires_api_keys: Tests requiring API keys")