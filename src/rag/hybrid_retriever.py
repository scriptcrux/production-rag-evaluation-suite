"""Hybrid Retriever combining BM25 keyword search and dense vector embeddings with RRF."""

import glob
import json
import os
import re
from typing import Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.core.exceptions import RetrievalException
from src.core.logger import get_logger
from src.domain.models import DocumentChunk
from src.rag.embeddings import EmbeddingService

logger = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    """Basic alphanumeric tokenizer for BM25 indexing."""
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """Enterprise Hybrid Search combining Dense Vector Search and BM25 Sparse Search
    via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        knowledge_base_dir: str = "data/knowledge_base",
        embedding_service: Optional[EmbeddingService] = None,
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        self.kb_dir = knowledge_base_dir
        self.embedding_service = embedding_service or EmbeddingService()
        self.rrf_k = rrf_k
        self.alpha = alpha

        self.documents: List[DocumentChunk] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self._is_indexed = False

    async def initialize(self):
        """Loads knowledge base files and builds vector and BM25 indices."""
        if self._is_indexed:
            return

        logger.info(f"Loading knowledge base from {self.kb_dir}")
        json_files = glob.glob(os.path.join(self.kb_dir, "*.json"))
        if not json_files:
            raise RetrievalException(f"No knowledge base JSON files found in {self.kb_dir}")

        all_chunks: List[DocumentChunk] = []
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        chunk = DocumentChunk(**item)
                        all_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        self.documents = all_chunks
        logger.info(f"Loaded {len(self.documents)} document chunks into index")

        # Build BM25 index
        corpus = [tokenize(f"{doc.ticker} {doc.section} {doc.content}") for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)

        # Build Vector index
        doc_texts = [f"{doc.ticker} {doc.company_name} - {doc.section}: {doc.content}" for doc in self.documents]
        embeddings = await self.embedding_service.get_embeddings(doc_texts)
        self.doc_embeddings = np.array(embeddings, dtype=np.float32)

        self._is_indexed = True
        logger.info("Hybrid index successfully initialized.")

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker_filter: Optional[str] = None,
        fiscal_year_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Performs hybrid retrieval using Reciprocal Rank Fusion."""
        if not self._is_indexed:
            await self.initialize()

        if not self.documents:
            return []

        # Step 1: BM25 Sparse Search
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        sparse_ranking = np.argsort(bm25_scores)[::-1]

        # Step 2: Dense Vector Search
        query_embedding = await self.embedding_service.get_embedding(query)
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity (both query and doc vectors are unit normalized)
        dense_scores = np.dot(self.doc_embeddings, query_vec)
        dense_ranking = np.argsort(dense_scores)[::-1]

        # Step 3: Reciprocal Rank Fusion (RRF)
        # RRF Score = alpha / (k + rank_dense) + (1 - alpha) / (k + rank_sparse)
        rrf_scores: Dict[int, float] = {}

        for rank, doc_idx in enumerate(dense_ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (self.alpha / (self.rrf_k + rank + 1))

        for rank, doc_idx in enumerate(sparse_ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + ((1.0 - self.alpha) / (self.rrf_k + rank + 1))

        # Sort documents by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

        results: List[DocumentChunk] = []
        for idx in sorted_indices:
            doc = self.documents[idx]

            # Apply metadata filters if provided
            if ticker_filter and doc.ticker.upper() != ticker_filter.upper():
                continue
            if fiscal_year_filter and doc.fiscal_year.upper() != fiscal_year_filter.upper():
                continue

            # Clone chunk with retrieval score
            doc_copy = doc.model_copy()
            doc_copy.score = float(rrf_scores[idx])
            results.append(doc_copy)

            if len(results) >= top_k:
                break

        return results

