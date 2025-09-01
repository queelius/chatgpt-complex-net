"""
Tests for TF-IDF embedding implementations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import gc

from embedding.tfidf_memory_efficient import MemoryEfficientTfidfVectorizer


class TestMemoryEfficientTfidfVectorizer:
    """Test memory-efficient TF-IDF implementation."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand human language",
            "Data science combines statistics and computer science"
        ]
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test vectorizer initialization."""
        vectorizer = MemoryEfficientTfidfVectorizer(
            max_features=100,
            min_df=2,
            max_df=0.9
        )
        
        assert vectorizer.max_features == 100
        assert vectorizer.min_df == 2
        assert vectorizer.max_df == 0.9
        assert vectorizer.vectorizer is None
    
    @pytest.mark.unit
    def test_default_initialization(self):
        """Test vectorizer with default parameters."""
        vectorizer = MemoryEfficientTfidfVectorizer()
        
        assert vectorizer.max_features == 5000
        assert vectorizer.min_df == 2
        assert vectorizer.max_df == 0.95
    
    @pytest.mark.unit
    def test_fit(self, sample_documents):
        """Test fitting the vectorizer."""
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=50)
        vectorizer.fit(sample_documents)
        
        assert vectorizer.vectorizer is not None
        assert hasattr(vectorizer.vectorizer, 'vocabulary_')
        assert len(vectorizer.vectorizer.vocabulary_) <= 50
    
    @pytest.mark.unit
    def test_fit_small_dataset(self):
        """Test fitting with very small dataset."""
        # With only 2 documents, min_df should be adjusted
        small_docs = ["document one has words", "document two has different words"]
        
        vectorizer = MemoryEfficientTfidfVectorizer(min_df=2)  # Will be adjusted
        vectorizer.fit(small_docs)
        
        # Should adjust min_df to be reasonable for dataset size
        assert vectorizer.vectorizer is not None
        # Should have created vocabulary
        assert len(vectorizer.vectorizer.vocabulary_) > 0
    
    @pytest.mark.unit
    def test_transform(self, sample_documents):
        """Test transforming documents."""
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=20)
        vectorizer.fit(sample_documents)
        
        # Transform single document
        result = vectorizer.transform(["machine learning and deep learning"])
        
        assert result is not None
        assert hasattr(result, 'toarray')  # Should be sparse matrix
        assert result.shape[0] == 1  # One document
        assert result.shape[1] == len(vectorizer.vectorizer.vocabulary_)
    
    @pytest.mark.unit
    def test_transform_batch(self, sample_documents):
        """Test batch transformation with memory efficiency."""
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=20)
        vectorizer.fit(sample_documents)
        
        # Transform in batches
        batch_results = list(vectorizer.transform_batch(
            sample_documents,
            batch_size=2
        ))
        
        assert len(batch_results) == len(sample_documents)
        
        # Check each result
        for idx, embedding in batch_results:
            assert idx < len(sample_documents)
            assert isinstance(embedding, list)
            assert len(embedding) == len(vectorizer.vectorizer.vocabulary_)
    
    @pytest.mark.unit
    def test_transform_batch_memory_cleanup(self, sample_documents):
        """Test that batch transformation cleans up memory."""
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=20)
        vectorizer.fit(sample_documents)
        
        # Mock gc.collect to verify it's called
        with patch('gc.collect') as mock_gc:
            list(vectorizer.transform_batch(sample_documents, batch_size=2))
            
            # gc.collect should be called after each batch
            assert mock_gc.call_count > 0
    
    @pytest.mark.unit
    def test_get_feature_names(self, sample_documents):
        """Test getting feature names."""
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=20)
        vectorizer.fit(sample_documents)
        
        features = vectorizer.get_feature_names()
        
        assert isinstance(features, np.ndarray)
        assert len(features) == len(vectorizer.vectorizer.vocabulary_)
        assert all(isinstance(f, str) for f in features)
    
    @pytest.mark.unit
    def test_empty_documents(self):
        """Test handling of empty documents."""
        docs = ["", "   ", "\n\n"]
        
        vectorizer = MemoryEfficientTfidfVectorizer()
        vectorizer.fit(["test document", "another document"])  # Fit on valid docs
        
        # Transform empty docs
        result = vectorizer.transform(docs)
        
        assert result is not None
        assert result.shape[0] == len(docs)
        # Empty documents should produce zero vectors
        assert result.sum() == 0
    
    @pytest.mark.unit
    def test_unicode_handling(self, sample_documents):
        """Test handling of unicode text."""
        unicode_docs = [
            "Thïs ïs ünicöde text",
            "文本处理 text processing",
            "Emoji test 😀 🎉"
        ]
        
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=20, min_df=1)
        vectorizer.fit(unicode_docs)
        
        assert vectorizer.vectorizer is not None
        # Should handle unicode without errors
        result = vectorizer.transform(unicode_docs)
        assert result.shape[0] == len(unicode_docs)
    
    @pytest.mark.unit
    def test_stop_words_removal(self):
        """Test that stop words are removed."""
        docs = [
            "the and is are were",  # All stop words
            "machine learning algorithm"  # Content words
        ]
        
        vectorizer = MemoryEfficientTfidfVectorizer(min_df=1)
        vectorizer.fit(docs)
        
        features = vectorizer.get_feature_names()
        
        # Common stop words should not be in features
        assert "the" not in features
        assert "and" not in features
        assert "is" not in features
        
        # Content words should be present
        assert "machine" in features or "learning" in features or "algorithm" in features
    
    @pytest.mark.unit
    def test_max_features_limit(self, sample_documents):
        """Test that max_features is respected."""
        max_features = 10
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=max_features)
        vectorizer.fit(sample_documents)
        
        assert len(vectorizer.vectorizer.vocabulary_) <= max_features
        
        # Transform should produce vectors of correct size
        result = vectorizer.transform(sample_documents)
        assert result.shape[1] <= max_features
    
    @pytest.mark.unit
    def test_batch_size_edge_cases(self, sample_documents):
        """Test edge cases for batch processing."""
        vectorizer = MemoryEfficientTfidfVectorizer()
        vectorizer.fit(sample_documents)
        
        # Batch size larger than dataset
        results = list(vectorizer.transform_batch(
            sample_documents,
            batch_size=100
        ))
        assert len(results) == len(sample_documents)
        
        # Batch size of 1
        results = list(vectorizer.transform_batch(
            sample_documents,
            batch_size=1
        ))
        assert len(results) == len(sample_documents)
    
    @pytest.mark.unit
    def test_consistency(self, sample_documents):
        """Test that transform gives consistent results."""
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=20)
        vectorizer.fit(sample_documents)
        
        # Transform same document multiple times
        doc = ["test document for consistency"]
        result1 = vectorizer.transform(doc).toarray()
        result2 = vectorizer.transform(doc).toarray()
        
        # Should give same result
        np.testing.assert_array_almost_equal(result1, result2)
    
    @pytest.mark.unit
    def test_batch_vs_regular_transform(self, sample_documents):
        """Test that batch transform gives same results as regular transform."""
        vectorizer = MemoryEfficientTfidfVectorizer(max_features=20)
        vectorizer.fit(sample_documents)
        
        # Regular transform
        regular_result = vectorizer.transform(sample_documents).toarray()
        
        # Batch transform
        batch_results = []
        for idx, embedding in vectorizer.transform_batch(sample_documents, batch_size=2):
            batch_results.append((idx, embedding))
        
        # Sort by index and compare
        batch_results.sort(key=lambda x: x[0])
        batch_array = np.array([emb for _, emb in batch_results])
        
        # Should be approximately equal
        np.testing.assert_array_almost_equal(regular_result, batch_array, decimal=5)