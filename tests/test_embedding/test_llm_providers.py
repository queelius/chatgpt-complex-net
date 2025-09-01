"""
Tests for LLM embedding providers.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import requests

from embedding.llm_providers import (
    EmbeddingProvider,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    HuggingFaceProvider,
    CohereProvider,
    get_embedding_provider
)


class TestOllamaProvider:
    """Test Ollama embedding provider."""
    
    @pytest.mark.unit
    def test_ollama_initialization(self):
        """Test Ollama provider initialization."""
        provider = OllamaProvider(host="http://test:11434", model="test-model")
        assert provider.host == "http://test:11434"
        assert provider.model == "test-model"
    
    @pytest.mark.unit
    def test_ollama_default_initialization(self):
        """Test Ollama provider with defaults."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://env:11434", "OLLAMA_MODEL": "env-model"}):
            provider = OllamaProvider()
            assert provider.host == "http://env:11434"
            assert provider.model == "env-model"
    
    @pytest.mark.unit
    @patch("requests.post")
    def test_ollama_get_embedding(self, mock_post):
        """Test getting embedding from Ollama."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        provider = OllamaProvider(host="http://test:11434", model="test-model")
        embedding = provider.get_embedding("test text")
        
        # Check the API was called correctly
        mock_post.assert_called_once_with(
            "http://test:11434/api/embeddings",
            json={"model": "test-model", "prompt": "test text"},
            timeout=30
        )
        
        # Check the embedding
        assert isinstance(embedding, np.ndarray)
        assert embedding.tolist() == [0.1, 0.2, 0.3]
    
    @pytest.mark.unit
    @patch("requests.post")
    def test_ollama_get_embedding_error(self, mock_post):
        """Test Ollama error handling."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        provider = OllamaProvider()
        with pytest.raises(requests.exceptions.RequestException):
            provider.get_embedding("test text")
    
    @pytest.mark.unit
    @patch("requests.post")
    def test_ollama_batch_embeddings(self, mock_post):
        """Test batch embedding generation."""
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        provider = OllamaProvider()
        embeddings = provider.get_embeddings_batch(["text1", "text2"])
        
        assert len(embeddings) == 2
        assert mock_post.call_count == 2  # Called once per text


class TestOpenAIProvider:
    """Test OpenAI embedding provider."""
    
    @pytest.mark.unit
    def test_openai_initialization_with_key(self):
        """Test OpenAI provider initialization with API key."""
        provider = OpenAIProvider(api_key="test-key", model="text-embedding-3-large")
        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-3-large"
        assert provider.embedding_dimension == 3072
    
    @pytest.mark.unit
    def test_openai_initialization_from_env(self):
        """Test OpenAI provider initialization from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIProvider()
            assert provider.api_key == "env-key"
            assert provider.model == "text-embedding-3-small"
            assert provider.embedding_dimension == 1536
    
    @pytest.mark.unit
    def test_openai_no_api_key_error(self):
        """Test OpenAI provider without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIProvider()
    
    @pytest.mark.unit
    @pytest.mark.requires_api_keys
    def test_openai_get_embedding(self):
        """Test getting embedding from OpenAI - requires openai package."""
        # Mock the openai module
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            import sys
            mock_openai = sys.modules['openai']
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            
            mock_embedding = Mock()
            mock_embedding.embedding = [0.1, 0.2, 0.3]
            mock_response = Mock()
            mock_response.data = [mock_embedding]
            mock_client.embeddings.create.return_value = mock_response
            
            # Re-import to get mocked version
            from embedding.llm_providers import OpenAIProvider
            
            provider = OpenAIProvider(api_key="test-key")
            embedding = provider.get_embedding("test text")
            
            # Check API call
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small",
                input="test text"
            )
            
            # Check embedding
            assert isinstance(embedding, np.ndarray)
            assert embedding.tolist() == [0.1, 0.2, 0.3]
    
    @pytest.mark.unit
    @pytest.mark.requires_api_keys
    def test_openai_batch_embeddings(self):
        """Test batch embedding generation - requires openai package."""
        # Mock the openai module
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            import sys
            mock_openai = sys.modules['openai']
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            
            mock_embeddings = [Mock(embedding=[0.1, 0.2, 0.3]), Mock(embedding=[0.4, 0.5, 0.6])]
            mock_response = Mock()
            mock_response.data = mock_embeddings
            mock_client.embeddings.create.return_value = mock_response
            
            # Re-import to get mocked version
            from embedding.llm_providers import OpenAIProvider
            
            provider = OpenAIProvider(api_key="test-key")
            embeddings = provider.get_embeddings_batch(["text1", "text2"])
            
            assert len(embeddings) == 2
            assert embeddings[0].tolist() == [0.1, 0.2, 0.3]
            assert embeddings[1].tolist() == [0.4, 0.5, 0.6]


class TestHuggingFaceProvider:
    """Test HuggingFace embedding provider."""
    
    @pytest.mark.unit
    def test_huggingface_initialization(self):
        """Test HuggingFace provider initialization."""
        provider = HuggingFaceProvider(api_key="test-key", model="test-model")
        assert provider.api_key == "test-key"
        assert provider.model == "test-model"
    
    @pytest.mark.unit
    def test_huggingface_no_api_key_error(self):
        """Test HuggingFace provider without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="HuggingFace API key required"):
                HuggingFaceProvider()
    
    @pytest.mark.unit
    @patch("requests.post")
    def test_huggingface_get_embedding(self, mock_post):
        """Test getting embedding from HuggingFace."""
        mock_response = Mock()
        mock_response.json.return_value = [[0.1, 0.2, 0.3]]  # Nested list response
        mock_post.return_value = mock_response
        
        provider = HuggingFaceProvider(api_key="test-key")
        embedding = provider.get_embedding("test text")
        
        # Check API call
        expected_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        mock_post.assert_called_once_with(
            expected_url,
            json={"inputs": "test text"},
            headers={"Authorization": "Bearer test-key"},
            timeout=30
        )
        
        # Check embedding
        assert isinstance(embedding, np.ndarray)
        assert embedding.tolist() == [0.1, 0.2, 0.3]


class TestProviderFactory:
    """Test the provider factory function."""
    
    @pytest.mark.unit
    def test_get_embedding_provider_ollama(self):
        """Test getting Ollama provider."""
        provider = get_embedding_provider("ollama", host="http://test:11434")
        assert isinstance(provider, OllamaProvider)
        assert provider.host == "http://test:11434"
    
    @pytest.mark.unit
    def test_get_embedding_provider_openai(self):
        """Test getting OpenAI provider."""
        provider = get_embedding_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "test-key"
    
    @pytest.mark.unit
    def test_get_embedding_provider_voyage(self):
        """Test getting Voyage/Anthropic provider."""
        provider = get_embedding_provider("voyage", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)
    
    @pytest.mark.unit
    def test_get_embedding_provider_unknown(self):
        """Test unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            get_embedding_provider("unknown")
    
    @pytest.mark.unit
    def test_get_embedding_provider_case_insensitive(self):
        """Test provider name is case insensitive."""
        provider = get_embedding_provider("OLLAMA")
        assert isinstance(provider, OllamaProvider)