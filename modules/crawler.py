"""
crawler.py - Academic Article Crawler Module
Fetches papers from arXiv and Semantic Scholar APIs.
"""

import requests
import time
import logging
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArXivCrawler:
    """Crawls academic papers from the arXiv API."""
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, max_results: int = 50, delay: float = 1.0):
        self.max_results = max_results
        self.delay = delay

    def fetch_papers(self, query: str, categories: List[str] = None) -> List[Dict]:
        """Fetch papers from arXiv matching the query."""
        cat_filter = ""
        if categories:
            cat_filter = " AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"
        params = {
            "search_query": f"all:{query}{cat_filter}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        papers = []
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            papers = self._parse_atom(response.text)
            logger.info(f"Retrieved {len(papers)} papers from arXiv.")
        except requests.RequestException as e:
            logger.error(f"ArXiv request failed: {e}")
        time.sleep(self.delay)
        return papers

    def _parse_atom(self, xml_text: str) -> List[Dict]:
        """Parse Atom XML response from arXiv API."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            id_el = entry.find("atom:id", ns)
            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            paper = {
                "title": title_el.text.strip() if title_el is not None else "",
                "abstract": summary_el.text.strip() if summary_el is not None else "",
                "authors": authors,
                "url": id_el.text.strip() if id_el is not None else "",
                "published": published_el.text if published_el is not None else "",
                "source": "arxiv",
            }
            papers.append(paper)
        return papers


class SemanticScholarCrawler:
    """Crawls papers from the Semantic Scholar API."""
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, max_results: int = 50, delay: float = 1.0, api_key: Optional[str] = None):
        self.max_results = max_results
        self.delay = delay
        self.headers = {"x-api-key": api_key} if api_key else {}

    def fetch_papers(self, query: str) -> List[Dict]:
        """Fetch papers from Semantic Scholar."""
        params = {
            "query": query,
            "limit": self.max_results,
            "fields": "title,abstract,authors,year,url,fieldsOfStudy",
        }
        papers = []
        try:
            response = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            for item in data.get("data", []):
                paper = {
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", "") or "",
                    "authors": [a["name"] for a in item.get("authors", [])],
                    "year": item.get("year"),
                    "url": item.get("url", ""),
                    "categories": item.get("fieldsOfStudy", []),
                    "source": "semantic_scholar",
                }
                papers.append(paper)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
        time.sleep(self.delay)
        return papers


class AcademicCrawler:
    """Unified crawler that aggregates from multiple sources."""

    def __init__(self, config: Dict = None):
        cfg = config or {}
        self.arxiv = ArXivCrawler(
            max_results=cfg.get("max_results", 30),
            delay=cfg.get("delay", 1.5),
        )
        self.semantic_scholar = SemanticScholarCrawler(
            max_results=cfg.get("max_results", 30),
            delay=cfg.get("delay", 1.5),
            api_key=cfg.get("semantic_scholar_api_key"),
        )

    def crawl(self, topics: List[str], categories: List[str] = None) -> List[Dict]:
        """Crawl papers for all topics from all sources, deduplicated."""
        all_papers = []
        seen_titles = set()
        for topic in topics:
            for paper in self.arxiv.fetch_papers(topic, categories):
                key = paper["title"].lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    all_papers.append(paper)
            for paper in self.semantic_scholar.fetch_papers(topic):
                key = paper["title"].lower().strip()
                if key not in seen_titles and paper["abstract"]:
                    seen_titles.add(key)
                    all_papers.append(paper)
        logger.info(f"Total unique papers collected: {len(all_papers)}")
        return all_papers


if __name__ == "__main__":
    crawler = AcademicCrawler()
    papers = crawler.crawl(
        topics=["large language models", "knowledge graphs", "neural embeddings"],
        categories=["cs.AI", "cs.LG", "cs.CL"],
    )
    print(f"Collected {len(papers)} papers.")
    for p in papers[:3]:
        print(f"  - {p['title']} ({p['source']})")
