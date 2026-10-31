"""Re-ranking module for fine-grained ranking of hybrid retrieval results."""

import re
from typing import List
from src.core.logger import get_logger
from src.domain.models import DocumentChunk

logger = get_logger(__name__)


class Reranker:
    """Scores and re-ranks candidate document chunks using cross-feature alignment,
    section authority weighting, and financial entity coverage.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def rerank(self, query: str, candidates: List[DocumentChunk]) -> List[DocumentChunk]:
        """Re-ranks retrieved candidates and returns the top-k highest quality chunks."""
        if not candidates:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w+\b", query_lower))

        scored_candidates = []
        for chunk in candidates:
            base_score = chunk.score or 0.5
            content_lower = chunk.content.lower()
            section_lower = chunk.section.lower()

            # 1. Exact phrase and word overlap scoring
            word_matches = sum(1 for w in query_words if w in content_lower)
            coverage_ratio = word_matches / max(len(query_words), 1)

            # 2. Financial entity bonus (numbers, percentages, currencies)
            numerical_terms = re.findall(r"\b(?:\$?\d+(?:\.\d+)?%?|\$\d+)\b", content_lower)
            number_density = min(len(numerical_terms) * 0.05, 0.25)

            # 3. Section authority boost (MD&A and Financial statements are prime sources)
            authority_boost = 0.0
            if "item 7" in section_lower or "md&a" in section_lower:
                authority_boost = 0.15
            elif "financial statements" in section_lower:
                authority_boost = 0.15
            elif "risk factors" in section_lower and ("risk" in query_lower or "threat" in query_lower):
                authority_boost = 0.20

            # Combined re-rank score
            final_score = (base_score * 0.4) + (coverage_ratio * 0.3) + number_density + authority_boost

            updated_chunk = chunk.model_copy()
            updated_chunk.score = round(final_score, 4)
            scored_candidates.append(updated_chunk)

        # Sort descending by re-rank score
        scored_candidates.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_candidates[: self.top_k]
