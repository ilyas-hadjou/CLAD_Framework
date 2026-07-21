"""
Embedding Engine Module
========================
Wrapper for SentenceTransformer (Template Vector Bank Creation)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Wrapper for SentenceTransformer to create Template Vector Bank.
    
    Responsibilities:
    - Embed template strings into 384-dim vectors
    - Support batch embedding for efficiency
    - Provide similarity search functionality
    
    Usage:
        engine = EmbeddingEngine()
        embeddings = engine.embed_templates(["generating <*>", "iar <*> dear <*>"])
        similarities = engine.compute_similarity(query_embedding, template_bank)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding engine.
        
        Args:
            model_name: SentenceTransformer model (384-dim recommended for speed)
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"✅ Embedding model loaded (dim={self.model.get_sentence_embedding_dimension()})")
    
    def embed_templates(self, templates: List[str]) -> np.ndarray:
        """
        Embed a list of template strings.
        
        Args:
            templates: List of template strings (e.g., ["generating <*>", ...])
        
        Returns:
            numpy array of shape (len(templates), embedding_dim)
        """
        logger.info(f"Embedding {len(templates)} templates...")
        embeddings = self.model.encode(
            templates,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        logger.info(f"✅ Embeddings created: shape={embeddings.shape}")
        return embeddings
    
    def embed_single(self, template: str) -> np.ndarray:
        """
        Embed a single template string.
        
        Args:
            template: Template string
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.model.encode([template], convert_to_numpy=True)[0]
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        template_bank: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and template bank.
        
        Args:
            query_embedding: Shape (embedding_dim,) or (batch, embedding_dim)
            template_bank: Shape (num_templates, embedding_dim)
        
        Returns:
            Similarity scores of shape (num_templates,) or (batch, num_templates)
        """
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        bank_norm = template_bank / np.linalg.norm(template_bank, axis=-1, keepdims=True)
        
        # Cosine similarity via dot product
        if query_norm.ndim == 1:
            similarities = np.dot(bank_norm, query_norm)
        else:
            similarities = np.dot(query_norm, bank_norm.T)
        
        return similarities
    
    def find_top_k(
        self,
        query_embedding: np.ndarray,
        template_bank: np.ndarray,
        k: int = 5
    ) -> tuple:
        """
        Find top-k most similar templates.
        
        Args:
            query_embedding: Query vector
            template_bank: Template bank vectors
            k: Number of top results
        
        Returns:
            Tuple of (indices, scores)
        """
        similarities = self.compute_similarity(query_embedding, template_bank)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices, top_k_scores
