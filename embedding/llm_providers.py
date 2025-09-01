"""
LLM embedding providers for various services.
"""

import os
import numpy as np
import requests
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        pass
    
    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts (batch processing)."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        pass
    
    @property
    @abstractmethod
    def max_text_length(self) -> int:
        """Return the maximum text length this provider can handle."""
        pass


class OllamaProvider(EmbeddingProvider):
    """Ollama local LLM embedding provider."""
    
    def __init__(self, host: str = None, model: str = None):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
        self._dimension = None
        self._max_length = 8192  # Default for most models
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API."""
        endpoint = f"{self.host}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text[:self.max_text_length]
        }
        
        try:
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if embedding is None:
                raise ValueError("No 'embedding' key found in response")
            
            result = np.array(embedding)
            if self._dimension is None:
                self._dimension = len(result)
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Ollama doesn't support batch, so process sequentially."""
        return [self.get_embedding(text) for text in texts]
    
    @property
    def embedding_dimension(self) -> int:
        if self._dimension is None:
            # Get a sample embedding to determine dimension
            test_embedding = self.get_embedding("test")
            self._dimension = len(test_embedding)
        return self._dimension
    
    @property
    def max_text_length(self) -> int:
        return self._max_length


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings provider."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.model = model or "text-embedding-3-small"
        self._dimension = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }.get(self.model, 1536)
        
        self._max_length = 8191
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API."""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        response = client.embeddings.create(
            model=self.model,
            input=text[:self.max_text_length]
        )
        
        return np.array(response.data[0].embedding)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """OpenAI supports batch embedding."""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        # Truncate texts to max length
        texts = [text[:self.max_text_length] for text in texts]
        
        response = client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        return [np.array(item.embedding) for item in response.data]
    
    @property
    def embedding_dimension(self) -> int:
        return self._dimension
    
    @property
    def max_text_length(self) -> int:
        return self._max_length


class AnthropicProvider(EmbeddingProvider):
    """Anthropic (Claude) embeddings provider via Voyage AI."""
    
    def __init__(self, api_key: str = None, model: str = None):
        # Note: Anthropic doesn't have native embeddings, typically use Voyage AI
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key required for Anthropic-style embeddings")
        
        self.model = model or "voyage-2"
        self._dimension = {
            "voyage-2": 1024,
            "voyage-large-2": 1536,
            "voyage-code-2": 1536
        }.get(self.model, 1024)
        
        self._max_length = 4000
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Voyage AI API."""
        endpoint = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": [text[:self.max_text_length]],
            "input_type": "document"
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return np.array(data["data"][0]["embedding"])
        except requests.exceptions.RequestException as e:
            logger.error(f"Voyage AI API error: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Voyage AI supports batch embedding."""
        endpoint = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Truncate texts and batch (Voyage AI has a limit)
        texts = [text[:self.max_text_length] for text in texts]
        
        payload = {
            "model": self.model,
            "input": texts,
            "input_type": "document"
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return [np.array(item["embedding"]) for item in data["data"]]
        except requests.exceptions.RequestException as e:
            logger.error(f"Voyage AI API error: {e}")
            raise
    
    @property
    def embedding_dimension(self) -> int:
        return self._dimension
    
    @property
    def max_text_length(self) -> int:
        return self._max_length


class HuggingFaceProvider(EmbeddingProvider):
    """HuggingFace Inference API embedding provider."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HuggingFace API key required")
        
        self.model = model or "sentence-transformers/all-MiniLM-L6-v2"
        self._dimension = None
        self._max_length = 512  # Default for most sentence transformers
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from HuggingFace Inference API."""
        endpoint = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text[:self.max_text_length]}
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            embedding = response.json()
            
            # Handle nested response structure
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]
            
            result = np.array(embedding)
            if self._dimension is None:
                self._dimension = len(result)
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Process texts sequentially (batch not well supported in inference API)."""
        return [self.get_embedding(text) for text in texts]
    
    @property
    def embedding_dimension(self) -> int:
        if self._dimension is None:
            test_embedding = self.get_embedding("test")
            self._dimension = len(test_embedding)
        return self._dimension
    
    @property
    def max_text_length(self) -> int:
        return self._max_length


class CohereProvider(EmbeddingProvider):
    """Cohere embeddings provider."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required")
        
        self.model = model or "embed-english-v3.0"
        self._dimension = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384
        }.get(self.model, 1024)
        
        self._max_length = 512
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Cohere API."""
        import cohere
        
        co = cohere.Client(self.api_key)
        
        response = co.embed(
            texts=[text[:self.max_text_length]],
            model=self.model,
            input_type="search_document"
        )
        
        return np.array(response.embeddings[0])
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Cohere supports batch embedding."""
        import cohere
        
        co = cohere.Client(self.api_key)
        
        texts = [text[:self.max_text_length] for text in texts]
        
        response = co.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"
        )
        
        return [np.array(emb) for emb in response.embeddings]
    
    @property
    def embedding_dimension(self) -> int:
        return self._dimension
    
    @property
    def max_text_length(self) -> int:
        return self._max_length


# Provider registry
PROVIDERS = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "voyage": AnthropicProvider,  # Alias for clarity
    "huggingface": HuggingFaceProvider,
    "cohere": CohereProvider
}


def get_embedding_provider(provider_name: str, **kwargs) -> EmbeddingProvider:
    """Factory function to get an embedding provider."""
    provider_class = PROVIDERS.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDERS.keys())}")
    
    return provider_class(**kwargs)