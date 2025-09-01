"""
Memory-efficient TF-IDF embedding model.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from scipy.sparse import save_npz, load_npz
import logging
import gc

logger = logging.getLogger(__name__)

class MemoryEfficientTfidfVectorizer:
    def __init__(self, max_features=5000, min_df=2, max_df=0.95):
        """
        Initialize TF-IDF with memory constraints.
        
        Args:
            max_features: Maximum vocabulary size (reduces memory usage)
            min_df: Minimum document frequency (removes rare terms)
            max_df: Maximum document frequency (removes too common terms)
        """
        # Adjust min_df for small datasets
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vectorizer = None
        
    def fit(self, documents):
        """Fit the TF-IDF vectorizer on documents."""
        # Adjust min_df for small datasets
        n_docs = len(documents)
        actual_min_df = min(self.min_df, max(1, n_docs // 10))  # At most 10% of docs
        
        self.vectorizer = SklearnTfidfVectorizer(
            max_features=self.max_features,
            min_df=actual_min_df,
            max_df=self.max_df,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        
        logger.info(f"Fitting TF-IDF with max_features={self.max_features}, min_df={actual_min_df}")
        self.vectorizer.fit(documents)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
    def transform_batch(self, documents, batch_size=100):
        """
        Transform documents in batches to save memory.
        Yields dense arrays for each batch.
        """
        n_docs = len(documents)
        
        for i in range(0, n_docs, batch_size):
            batch = documents[i:i+batch_size]
            # Get sparse matrix for batch
            sparse_batch = self.vectorizer.transform(batch)
            # Convert to dense and yield
            dense_batch = sparse_batch.toarray()
            
            for j, vec in enumerate(dense_batch):
                yield i + j, vec.tolist()
            
            # Explicitly free memory
            del sparse_batch
            del dense_batch
            gc.collect()
    
    def transform(self, documents):
        """Transform documents to TF-IDF vectors (returns sparse matrix)."""
        return self.vectorizer.transform(documents)
    
    def get_feature_names(self):
        """Get vocabulary feature names."""
        return self.vectorizer.get_feature_names_out()