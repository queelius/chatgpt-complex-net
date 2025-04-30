"""
tdidf_embedding_model.py

Provides a function to compute TF-IDF embeddings for text documents.
Unlike LLM embeddings, this uses classical TF-IDF (bag-of-words) vectors.
"""

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

class TfidfVectorizer:
    def __init__(self):
        self.vectorizer = None

    def fit(self, documents):
        """
        Fit the TF-IDF vectorizer on a list of documents.
        """
        self.vectorizer = SklearnTfidfVectorizer()
        self.vectorizer.fit(documents)

    def transform(self, documents):
        """
        Transform a list of documents into TF-IDF vectors.
        Returns a numpy array of shape (num_docs, vocab_size).
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet.")
        return self.vectorizer.transform(documents).toarray()

    def get_tfidf_embedding(self, doc):
        """
        Compute the TF-IDF embedding for a single document.
        Returns a numpy array.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet.")
        return self.vectorizer.transform([doc]).toarray()[0]

def get_tfidf_embeddings(texts, vectorizer=None):
    """
    Given a list of texts, return their TF-IDF embeddings as numpy arrays.
    If vectorizer is None, fit a new one.
    Returns (vectorizer, embeddings)
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(texts)
    embeddings = vectorizer.transform(texts)
    return vectorizer, embeddings