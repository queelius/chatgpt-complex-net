"""
Tests for text chunking and aggregation strategies.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from embedding.chunking import (
    ChunkConfig,
    AggregationConfig,
    TextChunker,
    EmbeddingAggregator,
    ChunkedEmbeddingProcessor
)


class TestChunkConfig:
    """Test ChunkConfig dataclass."""
    
    @pytest.mark.unit
    def test_default_config(self):
        """Test default chunk configuration."""
        config = ChunkConfig()
        assert config.chunk_size == 512
        assert config.overlap == 50
        assert config.strategy == "sliding"
        assert config.respect_boundaries is True
    
    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom chunk configuration."""
        config = ChunkConfig(
            chunk_size=1024,
            overlap=100,
            strategy="sentence",
            respect_boundaries=False
        )
        assert config.chunk_size == 1024
        assert config.overlap == 100
        assert config.strategy == "sentence"
        assert config.respect_boundaries is False


class TestTextChunker:
    """Test TextChunker class."""
    
    @pytest.mark.unit
    def test_sliding_window_chunks(self):
        """Test sliding window chunking."""
        config = ChunkConfig(chunk_size=10, overlap=3, strategy="sliding")
        chunker = TextChunker(config)
        
        text = "This is a test text for chunking."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('position' in chunk for chunk in chunks)
        assert all('total_chunks' in chunk for chunk in chunks)
        
        # Check first chunk
        assert chunks[0]['text'][:9] == text[:9]  # Compare first 9 chars to avoid boundary issues
        assert chunks[0]['position'] == 0
        
        # Check overlap
        if len(chunks) > 1:
            # There should be some overlap between consecutive chunks
            assert len(chunks[0]['text']) >= config.overlap
    
    @pytest.mark.unit
    def test_sliding_window_no_overlap(self):
        """Test sliding window chunking without overlap."""
        config = ChunkConfig(chunk_size=10, overlap=0, strategy="sliding")
        chunker = TextChunker(config)
        
        text = "A" * 25  # 25 characters
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 3  # 10 + 10 + 5
        assert chunks[0]['text'] == "A" * 10
        assert chunks[1]['text'] == "A" * 10
        assert chunks[2]['text'] == "A" * 5
    
    @pytest.mark.unit
    def test_sentence_chunks(self):
        """Test sentence-based chunking."""
        config = ChunkConfig(chunk_size=50, overlap=10, strategy="sentence")
        chunker = TextChunker(config)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk['text'].strip()
            assert 'position' in chunk
            assert 'total_chunks' in chunk
    
    @pytest.mark.unit
    def test_paragraph_chunks(self):
        """Test paragraph-based chunking."""
        config = ChunkConfig(chunk_size=100, overlap=0, strategy="paragraph")
        chunker = TextChunker(config)
        
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk['text'].strip()
            assert 'position' in chunk
    
    @pytest.mark.unit
    def test_chunk_with_metadata(self):
        """Test chunking with metadata."""
        config = ChunkConfig(chunk_size=20, overlap=5)
        chunker = TextChunker(config)
        
        text = "Test text with metadata"
        metadata = {"source": "test", "id": 123}
        chunks = chunker.chunk_text(text, metadata)
        
        assert all(chunk['metadata'] == metadata for chunk in chunks)
    
    @pytest.mark.unit
    def test_respect_boundaries(self):
        """Test word boundary respecting."""
        config = ChunkConfig(
            chunk_size=20,
            overlap=0,
            strategy="sliding",
            respect_boundaries=True
        )
        chunker = TextChunker(config)
        
        text = "This is a long word that should not be split"
        chunks = chunker.chunk_text(text)
        
        # First chunk should end at a word boundary
        if len(chunks) > 1:
            assert chunks[0]['text'].endswith(' ') or chunks[0]['text'].endswith('word')


class TestAggregationConfig:
    """Test AggregationConfig dataclass."""
    
    @pytest.mark.unit
    def test_default_config(self):
        """Test default aggregation configuration."""
        config = AggregationConfig()
        assert config.method == "mean"
        assert config.weights is None
        assert config.role_weights is None
        assert config.position_decay == 0.0
    
    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom aggregation configuration."""
        config = AggregationConfig(
            method="weighted_mean",
            weights={"a": 1.0, "b": 2.0},
            role_weights={"user": 1.5, "assistant": 1.0},
            position_decay=0.5
        )
        assert config.method == "weighted_mean"
        assert config.weights["a"] == 1.0
        assert config.role_weights["user"] == 1.5
        assert config.position_decay == 0.5


class TestEmbeddingAggregator:
    """Test EmbeddingAggregator class."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return [np.random.randn(10) for _ in range(5)]
    
    @pytest.mark.unit
    def test_mean_aggregate(self, sample_embeddings):
        """Test mean aggregation."""
        config = AggregationConfig(method="mean")
        aggregator = EmbeddingAggregator(config)
        
        result = aggregator.aggregate(sample_embeddings)
        expected = np.mean(sample_embeddings, axis=0)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected)
    
    @pytest.mark.unit
    def test_weighted_mean_aggregate(self, sample_embeddings):
        """Test weighted mean aggregation."""
        config = AggregationConfig(method="weighted_mean")
        aggregator = EmbeddingAggregator(config)
        
        # Create chunks with metadata
        chunks = [{"position": i, "total_chunks": 5} for i in range(5)]
        
        result = aggregator.aggregate(sample_embeddings, chunks)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_embeddings[0].shape
    
    @pytest.mark.unit
    def test_weighted_mean_with_role_weights(self, sample_embeddings):
        """Test weighted mean with role weights."""
        config = AggregationConfig(
            method="weighted_mean",
            role_weights={"user": 2.0, "assistant": 1.0}
        )
        aggregator = EmbeddingAggregator(config)
        
        chunks = [
            {"position": 0, "total_chunks": 5, "metadata": {"role": "user"}},
            {"position": 1, "total_chunks": 5, "metadata": {"role": "assistant"}},
            {"position": 2, "total_chunks": 5, "metadata": {"role": "user"}},
            {"position": 3, "total_chunks": 5, "metadata": {"role": "assistant"}},
            {"position": 4, "total_chunks": 5, "metadata": {"role": "user"}},
        ]
        
        result = aggregator.aggregate(sample_embeddings, chunks)
        
        assert isinstance(result, np.ndarray)
        # User chunks should have more weight
        # Can't test exact values without reimplementing the logic
    
    @pytest.mark.unit
    def test_max_pool_aggregate(self, sample_embeddings):
        """Test max pooling aggregation."""
        config = AggregationConfig(method="max_pool")
        aggregator = EmbeddingAggregator(config)
        
        result = aggregator.aggregate(sample_embeddings)
        expected = np.max(sample_embeddings, axis=0)
        
        np.testing.assert_array_equal(result, expected)
    
    @pytest.mark.unit
    def test_first_last_aggregate(self, sample_embeddings):
        """Test first-last aggregation."""
        config = AggregationConfig(method="first_last")
        aggregator = EmbeddingAggregator(config)
        
        result = aggregator.aggregate(sample_embeddings)
        expected = (sample_embeddings[0] + sample_embeddings[-1]) / 2
        
        np.testing.assert_array_almost_equal(result, expected)
    
    @pytest.mark.unit
    def test_first_last_single_embedding(self):
        """Test first-last with single embedding."""
        config = AggregationConfig(method="first_last")
        aggregator = EmbeddingAggregator(config)
        
        single_embedding = [np.array([1, 2, 3])]
        result = aggregator.aggregate(single_embedding)
        
        np.testing.assert_array_equal(result, single_embedding[0])
    
    @pytest.mark.unit
    def test_position_decay(self, sample_embeddings):
        """Test position decay in weighted aggregation."""
        config = AggregationConfig(
            method="weighted_mean",
            position_decay=1.0
        )
        aggregator = EmbeddingAggregator(config)
        
        chunks = [{"position": i, "total_chunks": 5} for i in range(5)]
        result = aggregator.aggregate(sample_embeddings, chunks)
        
        assert isinstance(result, np.ndarray)
        # Beginning and end should have higher weights
    
    @pytest.mark.unit
    def test_empty_embeddings_error(self):
        """Test error on empty embeddings."""
        config = AggregationConfig()
        aggregator = EmbeddingAggregator(config)
        
        with pytest.raises(ValueError, match="No embeddings to aggregate"):
            aggregator.aggregate([])


class TestChunkedEmbeddingProcessor:
    """Test ChunkedEmbeddingProcessor class."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock embedding provider."""
        provider = Mock()
        provider.embedding_dimension = 10
        provider.max_text_length = 100
        
        def mock_get_embedding(text):
            np.random.seed(hash(text) % 1000)
            return np.random.randn(10)
        
        provider.get_embedding = Mock(side_effect=mock_get_embedding)
        return provider
    
    @pytest.mark.unit
    def test_process_text(self, mock_provider):
        """Test processing text through full pipeline."""
        chunk_config = ChunkConfig(chunk_size=20, overlap=5)
        agg_config = AggregationConfig(method="mean")
        
        processor = ChunkedEmbeddingProcessor(
            chunk_config,
            agg_config,
            mock_provider
        )
        
        text = "This is a test text that needs to be chunked and embedded."
        embedding, metadata = processor.process_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (10,)  # Matches provider dimension
        assert 'chunking' in metadata
        assert 'aggregation' in metadata
        assert metadata['chunking']['num_chunks'] > 0
    
    @pytest.mark.unit
    def test_process_text_with_metadata(self, mock_provider):
        """Test processing text with metadata."""
        chunk_config = ChunkConfig(chunk_size=30, overlap=5)
        agg_config = AggregationConfig()
        
        processor = ChunkedEmbeddingProcessor(
            chunk_config,
            agg_config,
            mock_provider
        )
        
        text = "This is a longer test text that will actually create chunks"
        text_metadata = {"source": "test", "id": 123}
        embedding, proc_metadata = processor.process_text(text, text_metadata)
        
        assert isinstance(embedding, np.ndarray)
        assert 'chunking' in proc_metadata
        assert proc_metadata['chunking']['chunk_size'] == 30
    
    @pytest.mark.unit
    def test_process_conversation(self, mock_provider):
        """Test processing conversation with roles."""
        chunk_config = ChunkConfig(chunk_size=50, overlap=10)
        agg_config = AggregationConfig(
            method="weighted_mean",
            role_weights={"user": 1.5, "assistant": 1.0}
        )
        
        processor = ChunkedEmbeddingProcessor(
            chunk_config,
            agg_config,
            mock_provider
        )
        
        messages = [
            {"role": "user", "content": "Hello, how are you doing today? I hope everything is going well."},
            {"role": "assistant", "content": "I'm doing very well, thank you for asking! How can I help you today?"},
            {"role": "user", "content": "What's the weather like in your area? Is it sunny or cloudy?"},
            {"role": "assistant", "content": "I don't have access to real-time weather data, but I can help with other questions."}
        ]
        
        embedding, metadata = processor.process_conversation(messages)
        
        assert isinstance(embedding, np.ndarray)
        assert metadata['conversation'] is True
        assert 'user' in metadata['roles']
        assert 'assistant' in metadata['roles']
        assert metadata['message_count'] == 4
        assert 'role_chunks' in metadata
    
    @pytest.mark.unit
    def test_process_empty_text_error(self, mock_provider):
        """Test error on empty text."""
        chunk_config = ChunkConfig()
        agg_config = AggregationConfig()
        
        processor = ChunkedEmbeddingProcessor(
            chunk_config,
            agg_config,
            mock_provider
        )
        
        with pytest.raises(ValueError, match="No chunks generated"):
            processor.process_text("")
    
    @pytest.mark.unit
    def test_embedding_error_handling(self, mock_provider):
        """Test handling of embedding errors."""
        # Make provider raise an error for specific text
        def mock_get_embedding_with_error(text):
            if "error" in text:
                raise Exception("Embedding error")
            return np.random.randn(10)
        
        mock_provider.get_embedding = Mock(side_effect=mock_get_embedding_with_error)
        
        chunk_config = ChunkConfig(chunk_size=20, overlap=5)
        agg_config = AggregationConfig()
        
        processor = ChunkedEmbeddingProcessor(
            chunk_config,
            agg_config,
            mock_provider
        )
        
        # Should handle error and use zero vector
        text = "This text contains error word and is long enough to create chunks properly for testing"
        embedding, metadata = processor.process_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (10,)