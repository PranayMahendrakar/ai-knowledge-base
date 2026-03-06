"""
pipeline.py - Main Orchestration Pipeline
Coordinates all modules to run the full knowledge base update cycle:
  1. Crawl new papers
  2. Clean and extract concepts
  3. Generate embeddings
  4. Build/update knowledge graph
  5. Update search index
  6. Export visualization data
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Module imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from modules.crawler import AcademicCrawler
from modules.text_cleaner import PaperProcessor
from modules.embedding_engine import EmbeddingEngine
from modules.knowledge_graph_builder import KnowledgeGraphBuilder
from modules.search_engine import KnowledgeBaseSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

# Default paths
DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
PAPERS_PATH = DATA_DIR / "papers.json"
GRAPH_PATH = DOCS_DIR / "graph_data.json"
SEARCH_INDEX_PATH = DATA_DIR / "search_index.json"
CONFIG_PATH = Path("config.json")

# Default configuration
DEFAULT_CONFIG = {
    "topics": [
        "large language models",
        "knowledge graphs",
        "neural embeddings",
        "transformer architecture",
        "retrieval augmented generation",
        "graph neural networks",
    ],
    "arxiv_categories": ["cs.AI", "cs.LG", "cs.CL", "cs.IR"],
    "max_results_per_topic": 20,
    "crawl_delay": 2.0,
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.65,
    "max_edges_per_node": 10,
    "top_concepts": 20,
    "summary_sentences": 3,
    "incremental": True,
}


def load_config(config_path: str = None) -> Dict:
    """Load configuration from JSON file or use defaults."""
    path = Path(config_path or CONFIG_PATH)
    if path.exists():
        with open(path) as f:
            config = json.load(f)
        logger.info(f"Loaded config from {path}.")
        return {**DEFAULT_CONFIG, **config}
    logger.info("Using default configuration.")
    return DEFAULT_CONFIG


def load_existing_papers(papers_path: str = None) -> List[Dict]:
    """Load previously crawled papers."""
    path = Path(papers_path or PAPERS_PATH)
    if path.exists():
        with open(path) as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} existing papers from {path}.")
        return papers
    return []


def save_papers(papers: List[Dict], papers_path: str = None) -> None:
    """Save papers to JSON."""
    path = Path(papers_path or PAPERS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Remove heavy embedding data before saving raw papers
    lightweight = [{k: v for k, v in p.items() if k != "embedding"} for p in papers]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lightweight, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(lightweight)} papers to {path}.")


def run_pipeline(
    config: Dict = None,
    skip_crawl: bool = False,
    skip_embed: bool = False,
    incremental: bool = None,
) -> Dict:
    """
    Run the full knowledge base update pipeline.

    Args:
        config: Configuration dict (uses defaults if None)
        skip_crawl: Skip crawling and use existing papers
        skip_embed: Skip embedding generation (use existing embeddings)
        incremental: Override incremental mode from config

    Returns:
        Summary dict with run statistics
    """
    cfg = config or load_config()
    if incremental is not None:
        cfg["incremental"] = incremental

    start_time = datetime.utcnow()
    stats = {"start_time": start_time.isoformat(), "steps": {}}

    # ── Step 1: Crawl ─────────────────────────────────────────────────────────
    if skip_crawl:
        logger.info("Skipping crawl. Loading existing papers.")
        new_papers = []
    else:
        logger.info("Step 1: Crawling academic papers...")
        crawler = AcowledgeCrawler({
            "max_results": cfg["max_results_per_topic"],
            "delay": cfg["crawl_delay"],
        })
        new_papers = crawler.crawl(
            topics=cfg["topics"],
            categories=cfg.get("arxiv_categories"),
        )
        stats["steps"]["crawl"] = {"papers_fetched": len(new_papers)}
        logger.info(f"Crawled {len(new_papers)} new papers.")

    # Load existing papers for incremental mode
    existing_papers = load_existing_papers() if cfg.get("incremental") else []
    existing_urls = {p.get("url", "") for p in existing_papers}
    truly_new = [p for p in new_papers if p.get("url", "") not in existing_urls]
    all_raw_papers = existing_papers + truly_new
    logger.info(f"Total papers in corpus: {len(all_raw_papers)} ({len(truly_new)} new).")

    # ── Step 2: Clean and Extract Concepts ───────────────────────────────────
    logger.info("Step 2: Cleaning text and extracting concepts...")
    processor = PaperProcessor({
        "top_concepts": cfg.get("top_concepts", 20),
        "summary_sentences": cfg.get("summary_sentences", 3),
    })
    papers_to_process = truly_new if cfg.get("incremental") else all_raw_papers
    processed_new = processor.process_papers(papers_to_process)
    stats["steps"]["clean"] = {"papers_processed": len(processed_new)}

    # Merge: use existing processed papers + new ones
    if cfg.get("incremental") and existing_papers:
        all_processed = existing_papers + processed_new
    else:
        all_processed = processed_new

    save_papers(all_processed)

    # ── Step 3: Generate Embeddings ───────────────────────────────────────────
    if skip_embed:
        logger.info("Skipping embedding generation.")
        all_embedded = all_processed
    else:
        logger.info("Step 3: Generating embeddings...")
        engine = EmbeddingEngine(model_name=cfg.get("embedding_model", "all-MiniLM-L6-v2"))

        papers_needing_emb = [p for p in all_processed if not p.get("embedding")]
        if papers_needing_emb:
            embedded_new = engine.embed_papers(papers_needing_emb)
            # Merge with already-embedded papers
            has_emb = {p.get("url", p.get("title", "")): p for p in all_processed if p.get("embedding")}
            for p in embedded_new:
                has_emb[p.get("url", p.get("title", ""))] = p
            all_embedded = list(has_emb.values())
        else:
            all_embedded = all_processed

        stats["steps"]["embed"] = {"papers_embedded": len(papers_needing_emb)}

    # ── Step 4: Build/Update Knowledge Graph ─────────────────────────────────
    logger.info("Step 4: Building knowledge graph...")
    builder = KnowledgeGraphBuilder(
        similarity_threshold=cfg.get("similarity_threshold", 0.65),
        max_edges_per_node=cfg.get("max_edges_per_node", 10),
        graph_path=str(GRAPH_PATH),
    )

    if cfg.get("incremental") and GRAPH_PATH.exists():
        graph = builder.update(str(GRAPH_PATH), all_embedded)
    else:
        graph = builder.build(all_embedded)

    graph.save(str(GRAPH_PATH))
    graph_stats = graph.stats()
    stats["steps"]["graph"] = graph_stats
    logger.info(f"Graph: {graph_stats}")

    # ── Step 5: Update Search Index ───────────────────────────────────────────
    logger.info("Step 5: Updating search index...")
    search = KnowledgeBaseSearch()
    search.build(all_embedded, graph_path=str(GRAPH_PATH))
    search.save_index(str(SEARCH_INDEX_PATH))
    stats["steps"]["search"] = {"documents_indexed": len(all_embedded)}

    # ── Step 6: Export metadata for GitHub Pages ──────────────────────────────
    logger.info("Step 6: Exporting visualization metadata...")
    meta = {
        "updated_at": datetime.utcnow().isoformat(),
        "total_papers": len(all_embedded),
        "new_papers_this_run": len(truly_new),
        "graph_nodes": graph_stats["nodes"],
        "graph_edges": graph_stats["edges"],
        "topics": cfg["topics"],
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    end_time = datetime.utcnow()
    stats["end_time"] = end_time.isoformat()
    stats["duration_seconds"] = (end_time - start_time).total_seconds()
    stats["summary"] = meta

    logger.info(f"Pipeline complete in {stats['duration_seconds']:.1f}s.")
    logger.info(f"Summary: {meta}")
    return stats


# Alias for the typo above (AowledgeCrawler -> AcademicCrawler)
AowledgeCrawler = AcademicCrawler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Knowledge Base Pipeline")
    parser.add_argument("--config", type=str, help="Path to config.json")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip crawling step")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    parser.add_argument("--full-rebuild", action="store_true", help="Rebuild from scratch (non-incremental)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stats = run_pipeline(
        config=cfg,
        skip_crawl=args.skip_crawl,
        skip_embed=args.skip_embed,
        incremental=not args.full_rebuild,
    )

    print(json.dumps(stats, indent=2))
