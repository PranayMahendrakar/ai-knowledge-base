"""
text_cleaner.py - Text Cleaning and Preprocessing Module
Cleans raw academic text, extracts key concepts, and generates summaries.
"""

import re
import string
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Common stop words for filtering
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "this", "that", "these", "those", "it", "its", "as",
    "we", "our", "they", "their", "he", "she", "his", "her", "i", "my",
    "such", "also", "which", "when", "than", "then", "so", "if", "not",
    "more", "most", "other", "some", "paper", "method", "approach", "show",
    "propose", "use", "using", "based", "results", "model", "models",
}

# Academic domain keywords to keep even if short
DOMAIN_KEYWORDS = {
    "ai", "ml", "nlp", "dl", "cv", "gnn", "llm", "rag", "kg", "rl",
}


class TextCleaner:
    """Cleans and normalizes raw academic text."""

    def __init__(self, min_word_length: int = 3):
        self.min_word_length = min_word_length

    def clean(self, text: str) -> str:
        """
        Clean and normalize text.

        Steps: lowercase, remove URLs, remove special chars, normalize whitespace.
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove LaTeX-style math
        text = re.sub(r"\\[a-z]+\{[^}]*\}", " ", text)
        text = re.sub(r"\$[^$]+\$", " ", text)

        # Remove special characters but keep hyphens in compound words
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)

        # Normalize hyphens and whitespace
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Split cleaned text into tokens."""
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words and very short tokens."""
        return [
            t for t in tokens
            if (len(t) >= self.min_word_length or t in DOMAIN_KEYWORDS)
            and t not in STOP_WORDS
        ]

    def process(self, text: str) -> List[str]:
        """Full pipeline: clean -> tokenize -> remove stopwords."""
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        return self.remove_stopwords(tokens)


class KeyConceptExtractor:
    """Extracts key concepts from academic paper text."""

    def __init__(self, top_n: int = 20, ngram_size: int = 3):
        self.top_n = top_n
        self.ngram_size = ngram_size
        self.cleaner = TextCleaner()

    def extract_unigrams(self, text: str) -> List[Tuple[str, int]]:
        """Extract top single-word concepts by frequency."""
        tokens = self.cleaner.process(text)
        counts = Counter(tokens)
        return counts.most_common(self.top_n)

    def extract_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, int]]:
        """Extract top n-gram concepts."""
        tokens = self.cleaner.process(text)
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i + n]))
        counts = Counter(ngrams)
        return counts.most_common(self.top_n)

    def extract_concepts(self, title: str, abstract: str) -> Dict:
        """
        Extract key concepts from a paper's title and abstract.

        Title words get boosted weight (x3) since they are more representative.
        """
        # Weight title words more heavily
        title_text = (title + " ") * 3
        full_text = title_text + abstract

        unigrams = self.extract_unigrams(full_text)
        bigrams = self.extract_ngrams(full_text, n=2)

        # Combine and normalize
        all_concepts = {}
        for term, count in unigrams:
            all_concepts[term] = count

        for term, count in bigrams:
            # Only keep bigrams with meaningful frequency
            if count >= 2:
                all_concepts[term] = count

        # Sort by frequency
        sorted_concepts = sorted(all_concepts.items(), key=lambda x: x[1], reverse=True)

        return {
            "top_concepts": [c[0] for c in sorted_concepts[:self.top_n]],
            "concept_weights": dict(sorted_concepts[:self.top_n]),
            "unigrams": [u[0] for u in unigrams[:10]],
            "bigrams": [b[0] for b in bigrams[:10]],
        }


class SummaryGenerator:
    """Generates extractive summaries from paper abstracts."""

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
        self.cleaner = TextCleaner()

    def _sentence_score(self, sentence: str, keywords: List[str]) -> float:
        """Score a sentence by keyword density."""
        tokens = self.cleaner.process(sentence)
        if not tokens:
            return 0.0
        keyword_hits = sum(1 for t in tokens if t in keywords)
        return keyword_hits / len(tokens)

    def summarize(self, abstract: str, title: str = "") -> str:
        """
        Generate an extractive summary from the abstract.

        Selects sentences with highest keyword density.
        """
        if not abstract:
            return ""

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
        if not sentences:
            return abstract

        if len(sentences) <= self.max_sentences:
            return abstract

        # Get keywords from title + abstract
        keywords = self.cleaner.process(title + " " + abstract)

        # Score sentences
        scored = [(s, self._sentence_score(s, keywords)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences and reorder by original position
        top_sentences_set = {s for s, _ in scored[:self.max_sentences]}
        summary_sentences = [s for s in sentences if s in top_sentences_set]

        return " ".join(summary_sentences)


class PaperProcessor:
    """Full preprocessing pipeline for academic papers."""

    def __init__(self, config: Dict = None):
        cfg = config or {}
        self.concept_extractor = KeyConceptExtractor(
            top_n=cfg.get("top_concepts", 20),
        )
        self.summarizer = SummaryGenerator(
            max_sentences=cfg.get("summary_sentences", 3),
        )
        self.cleaner = TextCleaner()

    def process_paper(self, paper: Dict) -> Dict:
        """
        Process a single paper dict.

        Returns enriched paper with concepts, summary, and cleaned text.
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        concepts = self.concept_extractor.extract_concepts(title, abstract)
        summary = self.summarizer.summarize(abstract, title)
        cleaned_abstract = self.cleaner.clean(abstract)

        return {
            **paper,
            "concepts": concepts["top_concepts"],
            "concept_weights": concepts["concept_weights"],
            "summary": summary,
            "cleaned_abstract": cleaned_abstract,
        }

    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """Process a batch of papers."""
        processed = []
        for paper in papers:
            try:
                processed.append(self.process_paper(paper))
            except Exception as e:
                logger.error(f"Error processing paper '{paper.get('title', '')}': {e}")
                processed.append(paper)
        logger.info(f"Processed {len(processed)} papers.")
        return processed


if __name__ == "__main__":
    sample_paper = {
        "title": "Attention Is All You Need",
        "abstract": (
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks. We propose a new simple network architecture, "
            "the Transformer, based solely on attention mechanisms. Experiments on two machine "
            "translation tasks show these models to be superior in quality."
        ),
    }
    processor = PaperProcessor()
    result = processor.process_paper(sample_paper)
    print("Concepts:", result["concepts"][:10])
    print("Summary:", result["summary"])
