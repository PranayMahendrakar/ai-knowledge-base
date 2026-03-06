"""
embedding_engine.py - Embedding Generation Module
Generates dense vector embeddings for academic papers using local models.
Uses sentence-transformers (all-MiniLM-L6-v2) as the default local model.
Falls back to TF-IDF sparse embeddings if the neural model is unavailable.
"""

import os
import json
import logging
import hashlib
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt to import optional heavy dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using TF-IDF fallback.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


DEFAULT_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = Path("data/embedding_cache")


class EmbeddingCache:
    """Disk-based cache for embeddings to avoid recomputation."""

    def __init__(self, cache_dir: Union[str, Path] = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        """Generate a hash key for the text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding or None."""
        path = self.cache_dir / f"{self._key(text)}.npy"
        if path.exists():
            return np.load(str(path))
        return None

    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        path = self.cache_dir / f"{self._key(text)}.npy"
        np.save(str(path), embedding)

    def clear(self):
        """Clear all cached embeddings."""
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
        logger.info("Embedding cache cleared.")


class SentenceTransformerEngine:
    """Generates embeddings using a local sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, use_cache: bool = True):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.cache = EmbeddingCache() if use_cache else None
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dim}")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to an embedding vector."""
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        if self.cache:
            self.cache.set(text, embedding)

        return embedding

    def encode_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode a batch of texts efficiently."""
        # Check cache first
        results = {}
        to_encode = []
        to_encode_idx = []

        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    results[i] = cached
                else:
                    to_encode.append(text)
                    to_encode_idx.append(i)
        else:
            to_encode = texts
            to_encode_idx = list(range(len(texts)))

        if to_encode:
            embeddings = self.model.encode(
                to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for idx, (orig_idx, emb) in enumerate(zip(to_encode_idx, embeddings)):
                results[orig_idx] = emb
                if self.cache:
                    self.cache.set(texts[orig_idx], emb)

        return np.array([results[i] for i in range(len(texts))])

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings."""
        return float(np.dot(emb1, emb2))


class TFIDFEngine:
    """Sparse TF-IDF based embeddings as fallback when neural models unavailable."""

    def __init__(self, max_features: int = 10000):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        self.vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True, ngram_range=(1, 2))
        self.fitted = False
        self.dim = max_features

    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer on a corpus."""
        self.vectorizer.fit(texts)
        self.fitted = True
        logger.info(f"TF-IDF vectorizer fitted on {len(texts)} documents.")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to a normalized TF-IDF vector."""
        if not self.fitted:
            raise RuntimeError("Call fit() with your corpus before encoding.")
        vec = self.vectorizer.transform([text]).toarray()
        return normalize(vec)[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts."""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        matrix = self.vectorizer.transform(texts).toarray()
        return normalize(matrix)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity."""
        return float(np.dot(emb1, emb2))


class EmbeddingEngine:
    """
    Unified embedding engine that auto-selects the best available backend.
    Prefers SentenceTransformer (neural) over TF-IDF (sparse).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, use_cache: bool = True):
        self.backend_name = None
        self.engine = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.engine = SentenceTransformerEngine(model_name, use_cache=use_cache)
                self.backend_name = "sentence_transformers"
                self.dim = self.engine.dim
                logger.info("Using SentenceTransformer backend.")
                return
            except Exception as e:
                logger.warning(f"SentenceTransformer init failed: {e}. Falling back to TF-IDF.")

        if SKLEARN_AVAILABLE:
            self.engine = TFIDFEngine()
            self.backend_name = "tfidf"
            self.dim = 10000
            logger.info("Using TF-IDF backend.")
        else:
            raise RuntimeError("No embedding backend available. Install sentence-transformers or scikit-learn.")

    def fit_corpus(self, texts: List[str]):
        """Fit the engine on a corpus (required for TF-IDF backend)."""
        if self.backend_name == "tfidf":
            self.engine.fit(texts)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.engine.encode(text)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of texts."""
        if self.backend_name == "sentence_transformers":
            return self.engine.encode_batch(texts, batch_size=batch_size)
        return self.engine.encode_batch(texts)

    def embed_paper(self, paper: Dict) -> Dict:
        """
        Generate an embedding for a paper using title + abstract.
        Returns paper dict enriched with 'embedding' and 'embedding_dim' keys.
        """
        text = paper.get("title", "") + ". " + paper.get("abstract", "")
        embedding = self.encode(text)
        return {**paper, "embedding": embedding.tolist(), "embedding_dim": len(embedding)}

    def embed_papers(self, papers: List[Dict], batch_size: int = 32) -> List[Dict]:
        """Embed a list of papers."""
        texts = [p.get("title", "") + ". " + p.get("abstract", "") for p in papers]

        if self.backend_name == "tfidf" and not self.engine.fitted:
            self.engine.fit(texts)

        embeddings = self.encode_batch(texts, batch_size=batch_size)
        enriched = []
        for paper, emb in zip(papers, embeddings):
            enriched.append({**paper, "embedding": emb.tolist(), "embedding_dim": len(emb)})

        logger.info(f"Generated embeddings for {len(enriched)} papers.")
        return enriched

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        a = np.array(emb1)
        b = np.array(emb2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def save_embeddings(self, papers: List[Dict], output_path: str):
        """Save paper embeddings to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = []
        for p in papers:
            if "embedding" in p:
                data.append({
                    "title": p.get("title", ""),
                    "url": p.get("url", ""),
                    "embedding": p["embedding"],
                    "embedding_dim": p.get("embedding_dim", 0),
                })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"Saved {len(data)} embeddings to {output_path}.")

    @staticmethod
    def load_embeddings(input_path: str) -> List[Dict]:
        """Load embeddings from a JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} embeddings from {input_path}.")
        return data


if __name__ == "__main__":
    engine = EmbeddingEngine()
    papers = [
        {"title": "Attention Is All You Need", "abstract": "We propose the Transformer."},
        {"title": "BERT Pre-training", "abstract": "Bidirectional encoder representations."},
    ]
    embedded = engine.embed_papers(papers)
    sim = engine.similarity(embedded[0]["embedding"], embedded[1]["embedding"])
    print(f"Backend: {engine.backend_name}, Dim: {engine.dim}")
    print(f"Similarity between papers: {sim:.4f}")
