# AI Knowledge Base

> A continuously evolving, self-updating AI knowledge base that crawls academic papers, extracts concepts, generates semantic embeddings, and visualizes relationships as an interactive knowledge graph — updated weekly via GitHub Actions.

[![Update Knowledge Base](https://github.com/PranayMahendrakar/ai-knowledge-base/actions/workflows/update_knowledge_base.yml/badge.svg)](https://github.com/PranayMahendrakar/ai-knowledge-base/actions/workflows/update_knowledge_base.yml)
[![GitHub Pages](https://img.shields.io/badge/Live%20Graph-GitHub%20Pages-blue?logo=github)](https://pranaymahendrakar.github.io/ai-knowledge-base/)

## Live Demo

**[🔗 Interactive Knowledge Graph →](https://pranaymahendrakar.github.io/ai-knowledge-base/)**

The interactive graph lets you:
- **Click** any node to see paper details, abstracts, and key concepts
- **Drag** nodes to rearrange the layout
- **Zoom** with scroll wheel; pan by dragging the background
- **Search** papers and concepts using the search bar
- **Filter** by node type (papers vs. concepts) or edge type (similarity, discusses, shared concepts)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Weekly GitHub Actions                     │
│  (schedule: Monday 06:00 UTC + manual dispatch)             │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │         pipeline.py          │  ← Orchestrator
          └──────────────┬──────────────┘
                         │
     ┌───────────────────┼──────────────────────┐
     │                   │                      │
     ▼                   ▼                      ▼
┌─────────┐       ┌────────────┐       ┌──────────────┐
│ crawler │  →    │text_cleaner│  →    │  embedding   │
│  .py    │       │   .py      │       │  _engine.py  │
└─────────┘       └────────────┘       └──────────────┘
  arXiv API         Key concept          all-MiniLM-L6-v2
  Semantic          extraction           local neural model
  Scholar           BM25 scoring         (TF-IDF fallback)
                    Summaries
                         │
                         ▼
               ┌──────────────────┐
               │knowledge_graph   │
               │ _builder.py      │
               └────────┬─────────┘
                        │  graph_data.json
                ┌───────┴────────┐
                │                │
                ▼                ▼
        ┌─────────────┐  ┌──────────────┐
        │search_engine│  │  docs/       │
        │   .py       │  │  index.html  │
        └─────────────┘  └──────────────┘
          Hybrid search    D3.js visualization
          (BM25 + cosine)  GitHub Pages
```

---

## Modules

### `modules/crawler.py` — Academic Crawler
Fetches papers from two sources:
- **arXiv API** – searches by topic + category filters (e.g., `cs.AI`, `cs.LG`, `cs.CL`)
- **Semantic Scholar API** – additional papers with field-of-study metadata

Includes polite rate-limiting, deduplication by URL/title, and Atom XML parsing.

### `modules/text_cleaner.py` — Text Preprocessing
Three-stage cleaning pipeline:
1. **TextCleaner** – lowercases text, removes LaTeX math, URLs, and special characters
2. **KeyConceptExtractor** – extracts top unigram and bigram concepts weighted by title prominence (title tokens get 3× boost)
3. **SummaryGenerator** – extractive summarization by ranking sentences on keyword density

### `modules/embedding_engine.py` — Embedding Generation
Generates dense vector representations using local models — no external API calls required:
- **Primary**: `sentence-transformers` with `all-MiniLM-L6-v2` (384-dim, fast, accurate)
- **Fallback**: TF-IDF + L2 normalization (scikit-learn) when neural model is unavailable
- **Caching**: Disk-based MD5-keyed `.npy` cache to avoid recomputation across runs

### `modules/knowledge_graph_builder.py` — Graph Construction
Builds a structured graph with three node types and three edge types:

| Node Type | Description |
|-----------|-------------|
| **Paper** | Each crawled paper; carries title, abstract, authors, URL, concepts |
| **Concept** | Shared keywords appearing in ≥2 papers |
| **Author** | (Optional) Author nodes for co-authorship edges |

| Edge Type | Condition | Weight |
|-----------|-----------|--------|
| `similar_to` | Cosine similarity ≥ 0.65 | Cosine score |
| `discusses` | Paper contains concept | 1.0 |
| `shares_concept` | Papers share ≥3 concepts | shared_count / 10 |

Graph construction steps:
1. Create paper nodes from processed papers
2. Compute pairwise cosine similarities (O(n²), pruned to top-K per node)
3. Find shared-concept pairs and add weighted edges
4. Create concept nodes for concepts shared by ≥2 papers
5. Serialize to `docs/graph_data.json`

Incremental updates load the existing graph and only compute new edges for newly added papers.

### `modules/search_engine.py` — Hybrid Search
Multi-strategy search over the knowledge base:

| Mode | Description |
|------|-------------|
| **Keyword** | BM25 (k1=1.5, b=0.75) over title + abstract + concepts |
| **Semantic** | Cosine similarity against stored embeddings |
| **Hybrid** | Reciprocal Rank Fusion (RRF) of BM25 + semantic |
| **Graph** | BFS traversal from a seed node up to N hops |
| **Concept** | Filter by concept overlap score |

---

## Data Pipeline

```
Weekly cron trigger
       │
       ▼
1. CRAWL ──► arXiv + Semantic Scholar APIs
       │     Deduplication by URL/title
       │     Output: new_papers[]
       │
       ▼
2. CLEAN ──► TextCleaner + KeyConceptExtractor + SummaryGenerator
       │     Output: papers with concepts[], summary, cleaned_abstract
       │
       ▼
3. EMBED ──► sentence-transformers all-MiniLM-L6-v2
       │     Disk cache (data/embedding_cache/*.npy)
       │     Output: papers with embedding[384]
       │
       ▼
4. GRAPH ──► Similarity edges (cosine ≥ 0.65)
       │     Concept edges (shares ≥3 concepts)
       │     Concept nodes (shared by ≥2 papers)
       │     Output: docs/graph_data.json
       │
       ▼
5. INDEX ──► BM25 index + Semantic index
       │     Output: data/search_index.json
       │
       ▼
6. EXPORT ──► docs/metadata.json
             GitHub Actions commits & pushes
             GitHub Pages redeploys
```

---

## GitHub Actions Automation

The workflow `.github/workflows/update_knowledge_base.yml` runs every **Monday at 06:00 UTC** and can also be triggered manually with optional parameters:

| Input | Default | Description |
|-------|---------|-------------|
| `full_rebuild` | false | Rebuild from scratch (ignores existing graph) |
| `skip_crawl` | false | Skip crawling; process existing papers only |

After a successful pipeline run, the workflow automatically commits updated graph/metadata files and re-deploys GitHub Pages.

**Secrets required:**
- `SEMANTIC_SCHOLAR_API_KEY` (optional — higher rate limits)

---

## Graph Construction Methodology

### Node Scoring
Paper nodes are never scored or removed — all crawled papers are retained in the graph. Concept nodes are only created when a concept appears in ≥2 papers, preventing noise from hapax legomena.

### Similarity Threshold
The default cosine threshold of **0.65** was chosen empirically to balance:
- Enough edges for graph connectivity
- Avoidance of trivially-similar noise

### Edge Pruning
Each node retains at most **10 similarity edges** (configurable), keeping the graph navigable for large corpora. Only the highest-weight edges per node are kept.

### Incremental Updates
On each weekly run, the system:
1. Identifies papers not yet in the corpus by URL
2. Processes and embeds only the new papers
3. Recomputes similarity edges between new papers and all existing papers
4. Increments the graph version counter

This design keeps weekly update runtime low (minutes rather than hours) as the corpus grows.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/PranayMahendrakar/ai-knowledge-base.git
cd ai-knowledge-base

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python pipeline.py

# 4. Open the visualization
open docs/index.html

# 5. Search the knowledge base
python -c "
from modules.search_engine import KnowledgeBaseSearch
import json
with open('data/papers.json') as f: papers = json.load(f)
engine = KnowledgeBaseSearch()
engine.build(papers, graph_path='docs/graph_data.json')
results = engine.search('transformer attention', top_k=5, mode='keyword')
for r in results: print(r.rank, r.document['title'])
"
```

### Configuration
Edit `config.json` to customize topics, embedding model, similarity threshold, and more. See [Modules](#modules) for all available options.

---

## Project Structure

```
ai-knowledge-base/
├── modules/
│   ├── crawler.py              # Academic paper crawler
│   ├── text_cleaner.py         # Text cleaning & concept extraction
│   ├── embedding_engine.py     # Local neural embeddings
│   ├── knowledge_graph_builder.py  # Graph construction & update
│   └── search_engine.py        # Hybrid BM25 + semantic search
├── docs/
│   ├── index.html              # D3.js interactive visualization (GitHub Pages)
│   ├── graph_data.json         # Generated knowledge graph (auto-updated)
│   └── metadata.json           # Run metadata (auto-updated)
├── data/
│   ├── papers.json             # Processed paper corpus (auto-updated)
│   └── embedding_cache/        # Cached embeddings (gitignored)
├── .github/
│   └── workflows/
│       └── update_knowledge_base.yml  # Weekly automation
├── pipeline.py                 # Main orchestration script
├── config.json                 # Configuration
└── requirements.txt            # Python dependencies
```

---

## Technologies

| Layer | Technology |
|-------|-----------|
| Crawling | arXiv Atom API, Semantic Scholar REST API |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Fallback Embeddings | TF-IDF + scikit-learn |
| Graph Storage | JSON (custom Node/Edge schema) |
| Search | BM25 + Cosine Similarity + RRF fusion |
| Visualization | D3.js v7 (force-directed graph) |
| Hosting | GitHub Pages |
| Automation | GitHub Actions (cron schedule) |

---

*Built with Python · D3.js · GitHub Actions · GitHub Pages*
