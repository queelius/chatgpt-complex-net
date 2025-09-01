"""
Text chunking and embedding aggregation strategies.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 512  # Characters per chunk
    overlap: int = 50  # Overlap between chunks
    strategy: str = "sliding"  # sliding, sentence, paragraph
    respect_boundaries: bool = True  # Try to break at sentence/word boundaries
    
    
@dataclass
class AggregationConfig:
    """Configuration for embedding aggregation."""
    method: str = "mean"  # mean, weighted_mean, max_pool, first_last, pca
    weights: Optional[Dict[str, float]] = None  # For weighted aggregation
    role_weights: Optional[Dict[str, float]] = None  # Weight by role (user/assistant)
    position_decay: float = 0.0  # Decay factor for position-based weighting (0=no decay)
    

class TextChunker:
    """Handles text chunking with various strategies."""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text according to configuration.
        
        Returns list of chunks with metadata.
        """
        if self.config.strategy == "sliding":
            return self._sliding_window_chunks(text, metadata)
        elif self.config.strategy == "sentence":
            return self._sentence_chunks(text, metadata)
        elif self.config.strategy == "paragraph":
            return self._paragraph_chunks(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
    
    def _sliding_window_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create overlapping chunks using sliding window."""
        chunks = []
        text_len = len(text)
        stride = self.config.chunk_size - self.config.overlap
        
        for i in range(0, text_len, stride):
            chunk_end = min(i + self.config.chunk_size, text_len)
            chunk_text = text[i:chunk_end]
            
            # Try to break at word boundary if configured
            if self.config.respect_boundaries and i + self.config.chunk_size < text_len:
                last_space = chunk_text.rfind(' ')
                if last_space > self.config.chunk_size * 0.8:  # Only if we're not losing too much
                    chunk_text = chunk_text[:last_space]
                    chunk_end = i + last_space
            
            chunks.append({
                'text': chunk_text,
                'start': i,
                'end': chunk_end,
                'position': len(chunks),
                'total_chunks': None,  # Will be set after all chunks created
                'metadata': metadata or {}
            })
            
            # Stop if we've processed all text
            if chunk_end >= text_len:
                break
        
        # Update total chunks count
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
            
        return chunks
    
    def _sentence_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create chunks based on sentence boundaries."""
        # Simple sentence splitting (can be improved with spacy/nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'position': len(chunks),
                    'metadata': metadata or {}
                })
                
                # Start new chunk with overlap if configured
                if self.config.overlap > 0 and current_chunk:
                    # Keep last sentences for overlap
                    overlap_text = ' '.join(current_chunk[-2:])  # Keep last 2 sentences
                    if len(overlap_text) <= self.config.overlap * 2:
                        current_chunk = current_chunk[-2:]
                        current_length = len(overlap_text)
                    else:
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'position': len(chunks),
                'metadata': metadata or {}
            })
        
        # Update total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
            
        return chunks
    
    def _paragraph_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create chunks based on paragraph boundaries."""
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'position': len(chunks),
                    'metadata': metadata or {}
                })
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'position': len(chunks),
                'metadata': metadata or {}
            })
        
        # Update total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
            
        return chunks


class EmbeddingAggregator:
    """Aggregates multiple chunk embeddings into a single embedding."""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
    
    def aggregate(self, embeddings: List[np.ndarray], chunks: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Aggregate multiple embeddings into one.
        
        Args:
            embeddings: List of embedding vectors
            chunks: Optional chunk metadata for weighted aggregation
        """
        if not embeddings:
            raise ValueError("No embeddings to aggregate")
        
        embeddings = [np.array(e) for e in embeddings]
        
        if self.config.method == "mean":
            return self._mean_aggregate(embeddings)
        elif self.config.method == "weighted_mean":
            return self._weighted_mean_aggregate(embeddings, chunks)
        elif self.config.method == "max_pool":
            return self._max_pool_aggregate(embeddings)
        elif self.config.method == "first_last":
            return self._first_last_aggregate(embeddings)
        elif self.config.method == "pca":
            return self._pca_aggregate(embeddings)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.method}")
    
    def _mean_aggregate(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Simple mean of all embeddings."""
        return np.mean(embeddings, axis=0)
    
    def _weighted_mean_aggregate(self, embeddings: List[np.ndarray], chunks: List[Dict[str, Any]] = None) -> np.ndarray:
        """Weighted mean based on chunk metadata."""
        weights = self._calculate_weights(embeddings, chunks)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted average
        weighted_embeddings = [emb * w for emb, w in zip(embeddings, weights)]
        return np.sum(weighted_embeddings, axis=0)
    
    def _calculate_weights(self, embeddings: List[np.ndarray], chunks: List[Dict[str, Any]] = None) -> np.ndarray:
        """Calculate weights for each embedding based on configuration."""
        n = len(embeddings)
        weights = np.ones(n)
        
        if chunks:
            # Apply role-based weights
            if self.config.role_weights:
                for i, chunk in enumerate(chunks):
                    role = chunk.get('metadata', {}).get('role')
                    if role and role in self.config.role_weights:
                        weights[i] *= self.config.role_weights[role]
            
            # Apply position decay
            if self.config.position_decay > 0:
                for i, chunk in enumerate(chunks):
                    position = chunk.get('position', i)
                    total = chunk.get('total_chunks', n)
                    # Higher weight for beginning and end
                    distance_from_ends = min(position, total - position - 1)
                    decay_factor = np.exp(-self.config.position_decay * distance_from_ends)
                    weights[i] *= decay_factor
            
            # Apply custom weights
            if self.config.weights:
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk.get('metadata', {}).get('id')
                    if chunk_id and chunk_id in self.config.weights:
                        weights[i] *= self.config.weights[chunk_id]
        
        return weights
    
    def _max_pool_aggregate(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Max pooling across all embeddings."""
        return np.max(embeddings, axis=0)
    
    def _first_last_aggregate(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Average of first and last embeddings (good for long documents)."""
        if len(embeddings) == 1:
            return embeddings[0]
        return (embeddings[0] + embeddings[-1]) / 2
    
    def _pca_aggregate(self, embeddings: List[np.ndarray], n_components: int = 1) -> np.ndarray:
        """Use PCA to find principal direction of embeddings."""
        from sklearn.decomposition import PCA
        
        if len(embeddings) < 2:
            return embeddings[0]
        
        # Stack embeddings
        X = np.vstack(embeddings)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        # Use first principal component scaled by explained variance
        principal_direction = pca.components_[0]
        
        # Project embeddings onto principal direction and take mean
        projections = X @ principal_direction.reshape(-1, 1)
        mean_projection = np.mean(projections)
        
        # Reconstruct embedding
        reconstructed = mean_projection * principal_direction
        
        # Combine with mean for stability
        return 0.7 * reconstructed + 0.3 * np.mean(embeddings, axis=0)


class ChunkedEmbeddingProcessor:
    """High-level processor for chunked embeddings."""
    
    def __init__(self, 
                 chunk_config: ChunkConfig,
                 aggregation_config: AggregationConfig,
                 embedding_provider):
        self.chunker = TextChunker(chunk_config)
        self.aggregator = EmbeddingAggregator(aggregation_config)
        self.embedding_provider = embedding_provider
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process text through chunking, embedding, and aggregation.
        
        Returns:
            Tuple of (final_embedding, processing_metadata)
        """
        # Chunk the text
        chunks = self.chunker.chunk_text(text, metadata)
        
        if not chunks:
            raise ValueError("No chunks generated from text")
        
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Get embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            try:
                embedding = self.embedding_provider.get_embedding(chunk['text'])
                chunk_embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed chunk {chunk['position']}: {e}")
                # Use zero vector as fallback
                if chunk_embeddings:
                    embedding = np.zeros_like(chunk_embeddings[0])
                else:
                    # Try to get dimension from provider
                    embedding = np.zeros(self.embedding_provider.embedding_dimension)
                chunk_embeddings.append(embedding)
        
        # Aggregate embeddings
        final_embedding = self.aggregator.aggregate(chunk_embeddings, chunks)
        
        # Prepare metadata
        processing_metadata = {
            'chunking': {
                'strategy': self.chunker.config.strategy,
                'chunk_size': self.chunker.config.chunk_size,
                'overlap': self.chunker.config.overlap,
                'num_chunks': len(chunks)
            },
            'aggregation': {
                'method': self.aggregator.config.method,
                'role_weights': self.aggregator.config.role_weights,
                'position_decay': self.aggregator.config.position_decay
            }
        }
        
        return final_embedding, processing_metadata
    
    def process_conversation(self, messages: List[Dict[str, str]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a conversation with role-aware chunking.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
        """
        # Group messages by role for separate processing
        role_texts = {}
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role not in role_texts:
                role_texts[role] = []
            role_texts[role].append(content)
        
        # Process each role separately
        role_embeddings = {}
        role_metadata = {}
        
        for role, texts in role_texts.items():
            combined_text = '\n\n'.join(texts)
            chunks = self.chunker.chunk_text(combined_text, {'role': role})
            
            # Embed chunks
            chunk_embeddings = []
            for chunk in chunks:
                chunk['metadata']['role'] = role
                embedding = self.embedding_provider.get_embedding(chunk['text'])
                chunk_embeddings.append(embedding)
            
            # Store for role-based aggregation
            role_embeddings[role] = chunk_embeddings
            role_metadata[role] = chunks
        
        # Combine all embeddings with role weights
        all_embeddings = []
        all_chunks = []
        
        for role in role_embeddings:
            all_embeddings.extend(role_embeddings[role])
            all_chunks.extend(role_metadata[role])
        
        # Aggregate with role awareness
        final_embedding = self.aggregator.aggregate(all_embeddings, all_chunks)
        
        processing_metadata = {
            'conversation': True,
            'roles': list(role_texts.keys()),
            'message_count': len(messages),
            'role_chunks': {role: len(chunks) for role, chunks in role_metadata.items()},
            'chunking': {
                'strategy': self.chunker.config.strategy,
                'chunk_size': self.chunker.config.chunk_size,
                'overlap': self.chunker.config.overlap
            },
            'aggregation': {
                'method': self.aggregator.config.method,
                'role_weights': self.aggregator.config.role_weights
            }
        }
        
        return final_embedding, processing_metadata