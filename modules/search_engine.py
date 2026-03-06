"""
search_engine.py - Semantic Search Engine Module
Provides both semantic (embedding-based) and keyword (BM25-style) search
over the knowledge base. Supports graph-traversal-based exploration.
"""

import json
import math
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Lightweight BM25 text index for keyword search.
    Does not require external dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.idf_cache: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def _doc_text(self, doc: Dict) -> str:
        """Combine title + abstract + concepts for indexing."""
        title = doc.get("title", "")
        abstract = doc.get("abstract", doc.get("summary", ""))
        concepts = " ".join(doc.get("concepts", []))
        return f"{title} {title} {abstract} {concepts}"  # title doubled for boost

    def build(self, documents: List[Dict]) -> None:
        """Build the BM25 index from a list of document dicts."""
        self.corpus = documents
        self.doc_lengths = []

        term_doc_matrix = []
        for doc in documents:
            text = self._doc_text(doc)
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))
            term_counts = defaultdict(int)
            for t in tokens:
                term_counts[t] += 1
            term_doc_matrix.append(dict(term_counts))

        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        # Compute document frequencies
        self.doc_freqs = defaultdict(int)
        for term_counts in term_doc_matrix:
            for term in term_counts:
                self.doc_freqs[term] += 1

        self._term_doc_matrix = term_doc_matrix
        logger.info(f"BM25 index built: {len(documents)} docs, {len(self.doc_freqs)} terms.")

    def _idf(self, term: str) -> float:
        """Compute IDF score for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        n = len(self.corpus)
        df = self.doc_freqs.get(term, 0)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        self.idf_cache[term] = idf
        return idf

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against a document."""
        tokens = self._tokenize(query)
        term_counts = self._term_doc_matrix[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0
        for term in tokens:
            if term not in term_counts:
                continue
            tf = term_counts[term]
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_idx, score) tuples for top-k results."""
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SemanticIndex:
    """
    Embedding-based semantic search index.
    Stores normalized paper embeddings and performs cosine similarity search.
    """

    def __init__(self):
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self, documents: List[Dict]) -> None:
        """Build the index from documents that have 'embedding' keys."""
        docs_with_emb = [d for d in documents if d.get("embedding")]
        if not docs_with_emb:
            logger.warning("No embeddings found in documents. Semantic search will be unavailable.")
            return

        self.documents = docs_with_emb
        emb_matrix = np.array([d["embedding"] for d in docs_with_emb], dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = emb_matrix / norms
        logger.info(f"Semantic index built: {len(self.documents)} docs, dim={self.embeddings.shape[1]}.")

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_idx, cosine_similarity) for top-k results."""
        if self.embeddings is None:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = np.dot(self.embeddings, q)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


class SearchResult:
    """Represents a single search result."""

    def __init__(self, document: Dict, score: float, rank: int, match_type: str = "hybrid"):
        self.document = document
        self.score = score
        self.rank = rank
        self.match_type = match_type

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "match_type": self.match_type,
            "title": self.document.get("title", ""),
            "url": self.document.get("url", ""),
            "summary": self.document.get("summary", self.document.get("abstract", ""))[:200],
            "concepts": self.document.get("concepts", [])[:5],
            "authors": self.document.get("authors", [])[:3],
            "published": self.document.get("published", self.document.get("year", "")),
            "source": self.document.get("source", ""),
        }


class KnowledgeBaseSearch:
    """
    Hybrid search engine combining semantic and BM25 keyword search.
    Supports:
    - Semantic similarity search (if embeddings available)
    - BM25 keyword search
    - Hybrid fusion (Reciprocal Rank Fusion)
    - Graph-traversal search (find related papers via graph)
    - Concept-based filtering
    """

    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Weight for semantic score vs BM25 in hybrid mode (0=BM25 only, 1=semantic only)
        """
        self.alpha = alpha
        self.bm25 = BM25Index()
        self.semantic = SemanticIndex()
        self.documents: List[Dict] = []
        self.graph_data: Optional[Dict] = None
        self._built = False

    def build(self, documents: List[Dict], graph_path: str = None) -> None:
        """
        Build all search indices.

        Args:
            documents: List of processed paper dicts
            graph_path: Optional path to graph_data.json for graph traversal search
        """
        self.documents = documents
        self.bm25.build(documents)
        self.semantic.build(documents)

        if graph_path and Path(graph_path).exists():
            with open(graph_path, "r") as f:
                self.graph_data = json.load(f)
            logger.info(f"Loaded graph data from {graph_path}.")

        self._built = True
        logger.info(f"Search engine ready: {len(documents)} documents indexed.")

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """Fuse two ranked lists using Reciprocal Rank Fusion."""
        scores = defaultdict(float)

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] += (1 - self.alpha) / (k + rank + 1)

        for rank, (idx, _) in enumerate(semantic_results):
            scores[idx] += self.alpha / (k + rank + 1)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        query_embedding: List[float] = None,
        filter_categories: List[str] = None,
        filter_concepts: List[str] = None,
    ) -> List[SearchResult]:
        """
        Search the knowledge base.

        Args:
            query: Text query string
            top_k: Number of results to return
            mode: "hybrid" | "semantic" | "keyword"
            query_embedding: Pre-computed embedding for semantic search
            filter_categories: Optional list of categories to filter by
            filter_concepts: Optional list of concepts to filter by

        Returns:
            List of SearchResult objects
        """
        if not self._built:
            raise RuntimeError("Call build() before searching.")

        bm25_results = []
        semantic_results = []

        if mode in ("keyword", "hybrid"):
            bm25_results = self.bm25.search(query, top_k=top_k * 3)

        if mode in ("semantic", "hybrid") and query_embedding:
            semantic_results = self.semantic.search(query_embedding, top_k=top_k * 3)
        elif mode in ("semantic", "hybrid") and self.semantic.embeddings is not None:
            # Use BM25 results if no query embedding
            pass

        # Fuse results
        if mode == "hybrid" and bm25_results and semantic_results:
            fused = self._reciprocal_rank_fusion(bm25_results, semantic_results)
        elif mode == "semantic" and semantic_results:
            fused = [(idx, score) for idx, score in semantic_results]
        else:
            fused = [(idx, score) for idx, score in bm25_results]

        # Resolve to documents and apply filters
        results = []
        for idx, score in fused:
            if idx >= len(self.documents):
                continue
            doc = self.documents[idx]

            # Apply category filter
            if filter_categories:
                doc_cats = [c.lower() for c in doc.get("categories", [])]
                if not any(fc.lower() in doc_cats for fc in filter_categories):
                    continue

            # Apply concept filter
            if filter_concepts:
                doc_concepts = [c.lower() for c in doc.get("concepts", [])]
                if not any(fc.lower() in doc_concepts for fc in filter_concepts):
                    continue

            results.append(SearchResult(doc, score, rank=len(results) + 1, match_type=mode))
            if len(results) >= top_k:
                break

        return results

    def graph_search(self, paper_id: str, depth: int = 2) -> List[SearchResult]:
        """
        Explore the knowledge graph from a given paper node.

        Returns all papers reachable within 'depth' hops.
        """
        if not self.graph_data:
            logger.warning("No graph data loaded. Call build() with graph_path.")
            return []

        # Build adjacency from graph edges
        adjacency = defaultdict(set)
        for edge in self.graph_data.get("edges", []):
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])

        # BFS
        visited = {paper_id}
        frontier = {paper_id}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        visited.discard(paper_id)  # Remove starting node

        # Map to documents
        id_to_doc = {d.get("url", ""): d for d in self.documents}
        results = []
        for node_id in visited:
            doc = id_to_doc.get(node_id)
            if doc:
                results.append(SearchResult(doc, score=1.0, rank=len(results) + 1, match_type="graph"))

        return results

    def concept_search(self, concepts: List[str], top_k: int = 10) -> List[SearchResult]:
        """Return papers that discuss the given concepts (ranked by concept overlap)."""
        scored = []
        for doc in self.documents:
            doc_concepts = set(c.lower() for c in doc.get("concepts", []))
            query_concepts = set(c.lower() for c in concepts)
            overlap = len(doc_concepts & query_concepts)
            if overlap > 0:
                scored.append((doc, overlap / len(query_concepts)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(doc, score, rank=i + 1, match_type="concept")
            for i, (doc, score) in enumerate(scored[:top_k])
        ]

    def save_index(self, path: str) -> None:
        """Save document list for later reloading."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # Save without embeddings for smaller file size
            docs_no_emb = [{k: v for k, v in d.items() if k != "embedding"} for d in self.documents]
            json.dump(docs_no_emb, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved search index ({len(self.documents)} docs) to {path}.")


if __name__ == "__main__":
    # Demo
    papers = [
        {
            "title": "Attention Is All You Need",
            "abstract": "The Transformer uses attention mechanisms for sequence modeling.",
            "concepts": ["transformer", "attention", "sequence", "neural network"],
            "url": "https://arxiv.org/abs/1706.03762",
            "source": "arxiv",
        },
        {
            "title": "BERT: Pre-training Deep Bidirectional Transformers",
            "abstract": "BERT achieves state-of-the-art results on NLP tasks using pre-training.",
            "concepts": ["bert", "pre-training", "nlp", "transformer"],
            "url": "https://arxiv.org/abs/1810.04805",
            "source": "arxiv",
        },
        {
            "title": "Knowledge Graph Embeddings",
            "abstract": "We embed knowledge graph entities into continuous vector spaces.",
            "concepts": ["knowledge graph", "embedding", "entity", "relation"],
            "url": "https://arxiv.org/abs/1301.3666",
            "source": "arxiv",
        },
    ]

    engine = KnowledgeBaseSearch()
    engine.build(papers)

    results = engine.search("transformer attention mechanism", top_k=3, mode="keyword")
    print("Keyword results:")
    for r in results:
        print(f"  [{r.rank}] {r.document['title']} (score={r.score:.3f})")

    concept_results = engine.concept_search(["transformer", "nlp"])
    print("Concept results:")
    for r in concept_results:
        print(f"  [{r.rank}] {r.document['title']}")
