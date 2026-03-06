"""
knowledge_graph_builder.py - Knowledge Graph Construction Module
Builds and maintains a graph of paper nodes connected by:
  - Semantic similarity (embedding cosine similarity)
  - Shared key concepts
  - Co-authorship
  - Citation relationships

Graph is serialized to JSON for use by the visualization layer.
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

GRAPH_OUTPUT_PATH = Path("docs/graph_data.json")
SIMILARITY_THRESHOLD = 0.65
MAX_EDGES_PER_NODE = 10


class Node:
    """Represents a paper or concept node in the knowledge graph."""

    def __init__(
        self,
        node_id: str,
        label: str,
        node_type: str = "paper",  # "paper" | "concept" | "author"
        metadata: Dict = None,
    ):
        self.id = node_id
        self.label = label
        self.type = node_type
        self.metadata = metadata or {}
        self.embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            **self.metadata,
        }
        return d


class Edge:
    """Represents a relationship between two nodes."""

    def __init__(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        metadata: Dict = None,
    ):
        self.source = source
        self.target = target
        self.relation = relation  # "similar_to" | "shares_concept" | "authored_by" | "cites"
        self.weight = weight
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 4),
            **self.metadata,
        }


class KnowledgeGraph:
    """
    In-memory knowledge graph with nodes and edges.
    Can load from and save to JSON.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.metadata: Dict = {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "version": 1,
        }

    def add_node(self, node: Node) -> None:
        """Add or update a node."""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge (avoids exact duplicates)."""
        # Check for existing edge between same pair with same relation
        for e in self.edges:
            if e.source == edge.source and e.target == edge.target and e.relation == edge.relation:
                # Update weight if higher
                if edge.weight > e.weight:
                    e.weight = edge.weight
                return
        self.edges.append(edge)

    def get_neighbors(self, node_id: str) -> List[str]:
        """Return IDs of all neighboring nodes."""
        neighbors = set()
        for edge in self.edges:
            if edge.source == node_id:
                neighbors.add(edge.target)
            elif edge.target == node_id:
                neighbors.add(edge.source)
        return list(neighbors)

    def to_dict(self) -> Dict:
        """Serialize graph to a JSON-compatible dict."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": {
                **self.metadata,
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "updated_at": datetime.utcnow().isoformat(),
            },
        }

    def save(self, path: str = None) -> None:
        """Save graph to JSON file."""
        output_path = Path(path or GRAPH_OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Graph saved: {len(self.nodes)} nodes, {len(self.edges)} edges -> {output_path}")

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """Load a graph from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = cls()
        graph.metadata = data.get("metadata", {})

        for nd in data.get("nodes", []):
            node_id = nd.pop("id")
            label = nd.pop("label")
            node_type = nd.pop("type", "paper")
            node = Node(node_id, label, node_type, metadata=nd)
            graph.nodes[node_id] = node

        for ed in data.get("edges", []):
            edge = Edge(
                source=ed["source"],
                target=ed["target"],
                relation=ed.get("relation", "related"),
                weight=ed.get("weight", 1.0),
            )
            graph.edges.append(edge)

        logger.info(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges from {path}")
        return graph

    def stats(self) -> Dict:
        """Return basic graph statistics."""
        degree = defaultdict(int)
        for edge in self.edges:
            degree[edge.source] += 1
            degree[edge.target] += 1
        degrees = list(degree.values())
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "isolated_nodes": len(self.nodes) - len(degree),
        }


class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from processed papers with embeddings.

    Graph construction methodology:
    1. Create a node for each paper
    2. Create concept nodes for top shared concepts
    3. Add similarity edges based on embedding cosine similarity
    4. Add concept edges (paper -> concept)
    5. Add author nodes and edges
    6. Prune low-weight edges per node to keep graph navigable
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_edges_per_node: int = MAX_EDGES_PER_NODE,
        include_concept_nodes: bool = True,
        include_author_nodes: bool = False,
        graph_path: str = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
        self.include_concept_nodes = include_concept_nodes
        self.include_author_nodes = include_author_nodes
        self.graph_path = graph_path or str(GRAPH_OUTPUT_PATH)

    def _paper_id(self, paper: Dict) -> str:
        """Generate a stable ID for a paper."""
        url = paper.get("url", "")
        title = paper.get("title", "")
        return url if url else f"paper_{abs(hash(title)) % 100000}"

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        va = np.array(a)
        vb = np.array(b)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def _add_paper_nodes(self, graph: KnowledgeGraph, papers: List[Dict]) -> None:
        """Create paper nodes from processed papers."""
        for paper in papers:
            node_id = self._paper_id(paper)
            node = Node(
                node_id=node_id,
                label=paper.get("title", "Untitled"),
                node_type="paper",
                metadata={
                    "url": paper.get("url", ""),
                    "abstract": paper.get("summary", paper.get("abstract", ""))[:300],
                    "authors": paper.get("authors", [])[:3],
                    "published": paper.get("published", paper.get("year", "")),
                    "source": paper.get("source", ""),
                    "concepts": paper.get("concepts", [])[:10],
                    "categories": paper.get("categories", []),
                },
            )
            if "embedding" in paper:
                node.embedding = paper["embedding"]
            graph.add_node(node)

    def _add_similarity_edges(self, graph: KnowledgeGraph, papers: List[Dict]) -> None:
        """Add edges between papers with high embedding similarity."""
        papers_with_emb = [p for p in papers if "embedding" in p]
        n = len(papers_with_emb)
        logger.info(f"Computing similarity matrix for {n} papers...")

        # Track top-k edges per node
        edges_per_node = defaultdict(list)

        for i in range(n):
            for j in range(i + 1, n):
                p1 = papers_with_emb[i]
                p2 = papers_with_emb[j]
                sim = self._cosine_similarity(p1["embedding"], p2["embedding"])

                if sim >= self.similarity_threshold:
                    id1 = self._paper_id(p1)
                    id2 = self._paper_id(p2)
                    edges_per_node[id1].append((sim, id2))
                    edges_per_node[id2].append((sim, id1))

        # Keep only top-k edges per node
        for source_id, candidates in edges_per_node.items():
            candidates.sort(reverse=True)
            for weight, target_id in candidates[:self.max_edges_per_node]:
                edge = Edge(
                    source=source_id,
                    target=target_id,
                    relation="similar_to",
                    weight=weight,
                )
                graph.add_edge(edge)

        logger.info(f"Added {len(graph.edges)} similarity edges.")

    def _add_concept_nodes(self, graph: KnowledgeGraph, papers: List[Dict]) -> None:
        """Add concept nodes and paper->concept edges."""
        concept_papers = defaultdict(list)

        for paper in papers:
            paper_id = self._paper_id(paper)
            for concept in paper.get("concepts", [])[:10]:
                concept_papers[concept].append(paper_id)

        # Only create concept nodes shared by at least 2 papers
        for concept, paper_ids in concept_papers.items():
            if len(paper_ids) < 2:
                continue

            concept_id = f"concept_{concept.replace(' ', '_')}"
            if concept_id not in graph.nodes:
                node = Node(
                    node_id=concept_id,
                    label=concept,
                    node_type="concept",
                    metadata={"paper_count": len(paper_ids)},
                )
                graph.add_node(node)

            for paper_id in paper_ids:
                edge = Edge(
                    source=paper_id,
                    target=concept_id,
                    relation="discusses",
                    weight=1.0,
                )
                graph.add_edge(edge)

        concept_nodes = sum(1 for n in graph.nodes.values() if n.type == "concept")
        logger.info(f"Added {concept_nodes} concept nodes.")

    def _add_shared_concept_edges(self, graph: KnowledgeGraph, papers: List[Dict]) -> None:
        """Add edges between papers that share key concepts."""
        concept_to_papers = defaultdict(set)
        for paper in papers:
            paper_id = self._paper_id(paper)
            for concept in paper.get("concepts", [])[:15]:
                concept_to_papers[concept].add(paper_id)

        pair_shared = defaultdict(int)
        for concept, paper_ids in concept_to_papers.items():
            paper_list = list(paper_ids)
            for i in range(len(paper_list)):
                for j in range(i + 1, len(paper_list)):
                    key = tuple(sorted([paper_list[i], paper_list[j]]))
                    pair_shared[key] += 1

        for (id1, id2), shared_count in pair_shared.items():
            if shared_count >= 3:
                weight = min(1.0, shared_count / 10.0)
                edge = Edge(
                    source=id1,
                    target=id2,
                    relation="shares_concept",
                    weight=weight,
                    metadata={"shared_concepts": shared_count},
                )
                graph.add_edge(edge)

    def build(self, papers: List[Dict]) -> KnowledgeGraph:
        """
        Build a full knowledge graph from a list of processed papers.

        Args:
            papers: List of paper dicts with 'embedding' and 'concepts' keys

        Returns:
            KnowledgeGraph instance
        """
        graph = KnowledgeGraph()
        logger.info(f"Building knowledge graph from {len(papers)} papers...")

        self._add_paper_nodes(graph, papers)
        logger.info(f"Added {len(graph.nodes)} paper nodes.")

        self._add_similarity_edges(graph, papers)
        self._add_shared_concept_edges(graph, papers)

        if self.include_concept_nodes:
            self._add_concept_nodes(graph, papers)

        stats = graph.stats()
        logger.info(f"Graph stats: {stats}")
        return graph

    def update(self, existing_path: str, new_papers: List[Dict]) -> KnowledgeGraph:
        """
        Incrementally update an existing graph with new papers.

        Args:
            existing_path: Path to the existing graph JSON file
            new_papers: New papers to add

        Returns:
            Updated KnowledgeGraph
        """
        try:
            graph = KnowledgeGraph.load(existing_path)
            logger.info(f"Loaded existing graph with {len(graph.nodes)} nodes.")
        except FileNotFoundError:
            logger.info("No existing graph found. Building fresh.")
            graph = KnowledgeGraph()

        # Filter out already-present papers
        existing_ids = set(graph.nodes.keys())
        truly_new = [p for p in new_papers if self._paper_id(p) not in existing_ids]
        logger.info(f"{len(truly_new)} genuinely new papers to add.")

        if not truly_new:
            return graph

        # Add new paper nodes
        self._add_paper_nodes(graph, truly_new)

        # Recompute edges for new papers against all existing papers
        all_papers = [
            {"title": n.label, "embedding": n.embedding, **n.metadata}
            for n in graph.nodes.values()
            if n.type == "paper" and n.embedding
        ]

        self._add_similarity_edges(graph, all_papers)
        self._add_shared_concept_edges(graph, all_papers)

        if self.include_concept_nodes:
            self._add_concept_nodes(graph, all_papers)

        graph.metadata["version"] = graph.metadata.get("version", 1) + 1
        return graph


if __name__ == "__main__":
    # Demo build
    papers = [
        {
            "title": "Attention Is All You Need",
            "abstract": "We propose the Transformer architecture based on attention.",
            "url": "https://arxiv.org/abs/1706.03762",
            "concepts": ["transformer", "attention", "neural network", "sequence"],
            "embedding": [0.1, 0.9, 0.3, 0.7],
            "authors": ["Vaswani", "Shazeer"],
            "source": "arxiv",
        },
        {
            "title": "BERT: Pre-training Deep Bidirectional Transformers",
            "abstract": "We introduce BERT for language understanding.",
            "url": "https://arxiv.org/abs/1810.04805",
            "concepts": ["bert", "transformer", "pre-training", "language model"],
            "embedding": [0.15, 0.85, 0.35, 0.65],
            "authors": ["Devlin", "Chang"],
            "source": "arxiv",
        },
    ]
    builder = KnowledgeGraphBuilder(similarity_threshold=0.5)
    graph = builder.build(papers)
    graph.save("docs/graph_data.json")
    print(graph.stats())
