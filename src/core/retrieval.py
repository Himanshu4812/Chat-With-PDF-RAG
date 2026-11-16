"""Shared retrieval helpers: dedup, hybrid search, context formatting."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    """Stable hash for dedup of near-identical chunks."""
    return hashlib.sha256(text.strip().encode("utf-8", errors="ignore")).hexdigest()


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """Remove exact-duplicate chunks (common with MultiQuery 3x search)."""
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in docs:
        h = content_hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    if len(unique) != len(docs):
        logger.info(f"Deduped {len(docs)} -> {len(unique)} chunks")
    return unique


def format_context(docs: List[Document], max_chunks: int = 10) -> str:
    """Format chunks with structured source labels for citation.

    Example: [Source: file.pdf p.3 #0]
    Includes page_number + chunk_index when available so the LLM can
    cite precisely instead of just the filename.
    """
    parts: List[str] = []
    for doc in docs[:max_chunks]:
        name = doc.metadata.get("pdf_name") or doc.metadata.get("source_file") or "Unknown"
        page = doc.metadata.get("page_number", "?")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        parts.append(f"[Source: {name} p.{page} #{chunk_idx}]\n{doc.page_content}\n")
    return "\n---\n".join(parts)


def build_hybrid_retriever(
    vector_db: Any,
    docs_for_bm25: List[Document] | None,
    llm: Any,
    query_prompt: Any,
    k: int = 5,
    fetch_k: int = 20,
    use_mmr: bool = True,
    vector_weight: float = 0.7,
    enable_hybrid: bool = True,
    use_multi_query: bool = True,
) -> Any:
    """Build (optional MultiQuery ->) hybrid BM25+vector or vector-only retriever.

    Falls back gracefully when optional deps are missing:
    - BM25 needs langchain_community + rank_bm25; if unavailable, vector-only.
    - MMR needs Chroma support; if it fails, plain similarity.
    """
    # Base vector retriever with MMR for diversity when enabled
    try:
        if use_mmr:
            base_retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k},
            )
        else:
            base_retriever = vector_db.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.warning(f"MMR retriever failed, falling back to similarity: {e}")
        base_retriever = vector_db.as_retriever(search_kwargs={"k": k})

    base_for_multi = base_retriever

    if enable_hybrid and docs_for_bm25:
        try:
            from langchain_community.retrievers import BM25Retriever

            bm25 = BM25Retriever.from_documents(docs_for_bm25)
            bm25.k = k
            try:
                from langchain_classic.retrievers import EnsembleRetriever

                ensemble = EnsembleRetriever(
                    retrievers=[bm25, base_retriever],
                    weights=[1.0 - vector_weight, vector_weight],
                )
                base_for_multi = ensemble
                logger.info("Hybrid BM25+vector retriever enabled")
            except ImportError:
                logger.warning("EnsembleRetriever unavailable; using vector-only")
        except Exception as e:
            logger.warning(f"BM25 setup failed, using vector-only: {e}")

    if not use_multi_query:
        return base_for_multi

    # Multi-query expansion for recall (extra LLM call per query)
    try:
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        return MultiQueryRetriever.from_llm(
            retriever=base_for_multi,
            llm=llm,
            prompt=query_prompt,
        )
    except Exception as e:
        logger.warning(f"MultiQuery setup failed, using base retriever: {e}")
        return base_for_multi


def extract_sources(docs: List[Document], max_chunks: int = 10) -> List[dict]:
    """Build API source payload with page info for citations."""
    return [
        {
            "pdf_name": doc.metadata.get("pdf_name"),
            "pdf_id": doc.metadata.get("pdf_id"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "page_number": doc.metadata.get("page_number"),
        }
        for doc in docs[:max_chunks]
    ]
