import numpy as np
import os
from .llm_providers import get_embedding_provider

# Legacy configuration for backward compatibility
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.0.225:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")

# Create a default provider instance
_default_provider = None

def get_llm_embedding(text, provider_name="ollama", **kwargs):
    """
    Get an embedding for the input text using the specified provider.
    
    For backward compatibility, defaults to Ollama with legacy configuration.
    
    Args:
        text: The text to embed
        provider_name: Name of the provider to use (ollama, openai, anthropic, etc.)
        **kwargs: Additional configuration for the provider
    """
    global _default_provider
    
    # Use cached provider if no custom config provided
    if not kwargs and provider_name == "ollama" and _default_provider:
        return _default_provider.get_embedding(text)
    
    # Create provider with configuration
    if provider_name == "ollama" and not kwargs:
        # Use legacy configuration for backward compatibility
        kwargs = {"host": OLLAMA_HOST, "model": MODEL_NAME}
    
    provider = get_embedding_provider(provider_name, **kwargs)
    
    # Cache default Ollama provider
    if not kwargs and provider_name == "ollama" and not _default_provider:
        _default_provider = provider
    
    return provider.get_embedding(text)

