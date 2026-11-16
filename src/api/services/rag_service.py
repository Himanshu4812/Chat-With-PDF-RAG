"""RAG query service."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import ollama
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

try:
    from ..database import PDFMetadata, ChatSession, ChatMessage
    from ..config import settings
    from ...core.llm import LLMManager
    from ...core.retrieval import (
        deduplicate_docs,
        format_context,
        build_hybrid_retriever,
        extract_sources,
    )
    from .generation import (
        StreamingChatProvider,
        get_streaming_provider,
        OllamaStreamingProvider,
        GroqStreamingProvider,
        ProviderError,
        ProviderTimeoutError,
    )
except (ImportError, ValueError):
    from src.api.database import PDFMetadata, ChatSession, ChatMessage
    from src.api.config import settings
    from src.core.llm import LLMManager
    from src.core.retrieval import (
        deduplicate_docs,
        format_context,
        build_hybrid_retriever,
        extract_sources,
    )
    from src.api.services.generation import (
        StreamingChatProvider,
        get_streaming_provider,
        OllamaStreamingProvider,
        GroqStreamingProvider,
        ProviderError,
        ProviderTimeoutError,
    )

logger = logging.getLogger(__name__)

# Chroma shares process-wide state (tenant/client). Concurrent first-time
# creation of handles for different collections races and raises
# "Could not connect to tenant default_tenant" / Rust binding errors.
_chroma_lock = threading.RLock()


@lru_cache(maxsize=1)
def _get_embeddings() -> OllamaEmbeddings:
    """Shared embedding client (avoids per-query client creation)."""
    return OllamaEmbeddings(model=settings.EMBEDDING_MODEL)


@lru_cache(maxsize=64)
def _get_vector_db_cached(collection_name: str) -> Chroma:
    return Chroma(
        persist_directory=settings.VECTOR_DB_DIR,
        embedding_function=_get_embeddings(),
        collection_name=collection_name,
    )


def _get_vector_db(collection_name: str) -> Chroma:
    """Shared, thread-safe Chroma handle per collection."""
    with _chroma_lock:
        return _get_vector_db_cached(collection_name)


def _build_llm_manager(model: str) -> LLMManager:
    return LLMManager(
        model_name=model,
        num_ctx=settings.NUM_CTX,
        num_predict=settings.NUM_PREDICT,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        temperature=settings.LLM_TEMPERATURE,
    )


def _record_ollama_durations(timings: Dict[str, Any], source: Any) -> None:
    """Record Ollama's nanosecond durations as seconds plus tok/s."""
    if source is None:
        return
    for key in (
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
        "total_duration",
    ):
        value = (
            source.get(key)
            if isinstance(source, dict)
            else getattr(source, key, None)
        )
        if isinstance(value, (int, float)):
            timings[key.replace("_duration", "_seconds")] = round(
                value / 1e9, 2
            )
    eval_seconds = timings.get("eval_seconds")
    eval_count = timings.get("eval_count")
    if eval_seconds and eval_count:
        timings["eval_tokens_per_second"] = round(eval_count / eval_seconds, 1)


@dataclass
class StreamEvent:
    """One event emitted by RAGService.stream_query."""

    event: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedQuery:
    """Retrieval result that can feed sync or streaming generation."""

    mode: str
    messages: List[Dict[str, str]]
    sources: List[Dict[str, Any]]
    reasoning_steps: List[str]
    timings: Dict[str, Any]
    pdfs_queried: int
    use_thinking: bool
    question: str
    model: str
    llm_manager: Optional[LLMManager] = None


class RAGService:
    """Service for RAG operations."""

    def __init__(self):
        """Initialize RAG service."""
        self.persist_directory = settings.VECTOR_DB_DIR

    def prepare_query(
        self,
        question: str,
        model: str,
        pdf_ids: Optional[List[str]],
        db: Session,
    ) -> PreparedQuery:
        """Retrieve context and build the final chat messages.

        Semantics of pdf_ids:
        - None: query all indexed PDFs (benchmark/API default)
        - []: no PDF selected -> answer directly without RAG
        - [id, ...]: query only the selected PDFs
        """
        t_start = time.perf_counter()
        reasoning_steps: List[str] = []
        timings: Dict[str, Any] = {
            "num_ctx": settings.NUM_CTX,
            "num_predict": settings.NUM_PREDICT,
            "multi_query": settings.USE_MULTI_QUERY,
            "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
        }
        llm_manager = _build_llm_manager(model)
        use_thinking = LLMManager.is_thinking_model(model)

        def _direct(mode: str, steps: List[str], pdfs_queried: int) -> PreparedQuery:
            timings["retrieval_seconds"] = round(
                time.perf_counter() - t_start, 2
            )
            return PreparedQuery(
                mode=mode,
                messages=[{"role": "user", "content": question}],
                sources=[],
                reasoning_steps=steps,
                timings=timings,
                pdfs_queried=pdfs_queried,
                use_thinking=False,
                question=question,
                model=model,
                llm_manager=llm_manager,
            )

        if pdf_ids is not None and len(pdf_ids) == 0:
            return _direct(
                "direct",
                [
                    "💬 No PDFs selected. Responding directly with the model "
                    "(no document context)."
                ],
                0,
            )

        query = db.query(PDFMetadata)
        if pdf_ids is not None:
            query = query.filter(PDFMetadata.pdf_id.in_(pdf_ids))
        pdfs = query.all()

        if not pdfs:
            return _direct(
                "direct",
                [
                    "💬 No indexed PDFs found. Responding directly with the "
                    "model (no document context)."
                ],
                0,
            )

        reasoning_steps.append(
            f"📚 Searching across {len(pdfs)} PDF(s): "
            f"{', '.join([p.name for p in pdfs])}"
        )
        reasoning_steps.append(f"🤖 Using model: {model}")

        query_prompt = llm_manager.get_query_prompt()
        if settings.USE_MULTI_QUERY:
            reasoning_steps.append("🔍 Generating alternative search queries...")
        else:
            reasoning_steps.append("🔍 Retrieving relevant chunks...")

        t_retrieval_start = time.perf_counter()
        all_docs = []
        k = settings.RETRIEVAL_K
        max_chunks = settings.RETRIEVAL_MAX_CHUNKS

        for pdf in pdfs:
            try:
                with _chroma_lock:
                    vector_db = _get_vector_db(pdf.collection_name)

                    # BM25 needs the stored chunks; opt-in because reading
                    # the whole collection costs time on every query.
                    bm25_docs = None
                    if settings.ENABLE_HYBRID_SEARCH:
                        stored = vector_db.get(
                            include=["documents", "metadatas"]
                        )
                        raw_docs = stored.get("documents") or []
                        raw_metas = stored.get("metadatas") or []
                        from langchain_core.documents import Document as LCDoc

                        bm25_docs = [
                            LCDoc(
                                page_content=t,
                                metadata=m
                                or {
                                    "pdf_name": pdf.name,
                                    "pdf_id": pdf.pdf_id,
                                },
                            )
                            for t, m in zip(raw_docs, raw_metas)
                            if t
                        ] or None

                    retriever = build_hybrid_retriever(
                        vector_db=vector_db,
                        docs_for_bm25=bm25_docs,
                        llm=llm_manager.llm,
                        query_prompt=query_prompt,
                        k=k,
                        fetch_k=settings.RETRIEVAL_FETCH_K,
                        use_mmr=settings.USE_MMR,
                        vector_weight=settings.HYBRID_VECTOR_WEIGHT,
                        enable_hybrid=settings.ENABLE_HYBRID_SEARCH,
                        use_multi_query=settings.USE_MULTI_QUERY,
                    )

                    reasoning_steps.append(f"📄 Retrieving from: {pdf.name}")
                    docs = retriever.invoke(question)

                for doc in docs:
                    if "pdf_name" not in doc.metadata:
                        doc.metadata["pdf_name"] = pdf.name
                    if "pdf_id" not in doc.metadata:
                        doc.metadata["pdf_id"] = pdf.pdf_id
                all_docs.extend(docs)
                reasoning_steps.append(
                    f"✅ Found {len(docs)} relevant chunks in {pdf.name}"
                )
            except Exception as e:
                reasoning_steps.append(
                    f"⚠️ Error retrieving from {pdf.name}: {str(e)}"
                )
                logger.warning(f"Error retrieving from {pdf.name}: {e}")

        # Dedup across multi-PDF (and multi-query), then cap
        all_docs = deduplicate_docs(all_docs)
        reasoning_steps.append(f"📊 Total unique chunks retrieved: {len(all_docs)}")

        timings["retrieval_seconds"] = round(
            time.perf_counter() - t_retrieval_start, 2
        )

        if not all_docs:
            reasoning_steps.append(
                "⚠️ No relevant chunks found. Responding directly with the "
                "model instead."
            )
            timings["context_chunks"] = 0
            timings["context_chars"] = 0
            return _direct("rag_fallback", reasoning_steps, len(pdfs))

        formatted_context = format_context(all_docs, max_chunks=max_chunks)
        reasoning_steps.append(
            f"🔗 Using top {min(len(all_docs), max_chunks)} chunks for context"
        )
        timings["context_chunks"] = min(len(all_docs), max_chunks)
        timings["context_chars"] = len(formatted_context)

        if use_thinking:
            reasoning_steps.append(
                "🧠 Using thinking-enabled model with chain-of-thought "
                "reasoning..."
            )
            cot_system_message = llm_manager.get_cot_system_message(
                formatted_context
            )
            messages = [
                {"role": "system", "content": cot_system_message},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        "Think step-by-step and provide a detailed answer with "
                        "source citations."
                    ),
                },
            ]
        else:
            prompt = llm_manager.get_rag_prompt()
            lc_messages = prompt.format_messages(
                context=formatted_context, question=question
            )
            messages = [
                {
                    "role": _role_value(m),
                    "content": m.content
                    if isinstance(m.content, str)
                    else str(m.content),
                }
                for m in lc_messages
            ]

        reasoning_steps.append("💭 Generating answer with source citations...")

        return PreparedQuery(
            mode="rag",
            messages=messages,
            sources=extract_sources(all_docs, max_chunks=max_chunks),
            reasoning_steps=reasoning_steps,
            timings=timings,
            pdfs_queried=len(pdfs),
            use_thinking=use_thinking,
            question=question,
            model=model,
            llm_manager=llm_manager,
        )

    def _generate_sync(
        self,
        prepared: PreparedQuery,
        mode: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        """Generate a full (non-streaming) answer with provider fallback."""
        t_generation_start = time.perf_counter()
        timings = prepared.timings
        mode_val = (mode or getattr(settings, "RAG_MODE", "normal")).lower()

        # Determine primary and fallback providers
        primary_name = (
            provider.lower()
            if provider
            else ("groq" if mode_val == "presentation" else "ollama")
        )
        fallback_name = (
            None
            if mode_val == "offline"
            else ("ollama" if primary_name == "groq" else "groq")
        )

        def _call_ollama() -> str:
            llm_manager = prepared.llm_manager or _build_llm_manager(prepared.model)
            llm = llm_manager.llm
            if prepared.use_thinking:
                try:
                    ollama_response = ollama.chat(
                        model=prepared.model,
                        messages=prepared.messages,
                        think=True,
                        stream=False,
                        keep_alive=settings.OLLAMA_KEEP_ALIVE,
                        options={
                            "num_ctx": settings.NUM_CTX,
                            "num_predict": settings.NUM_PREDICT,
                            "temperature": settings.LLM_TEMPERATURE,
                        },
                    )
                    thinking_text = getattr(
                        ollama_response.message, "thinking", None
                    )
                    if thinking_text:
                        prepared.reasoning_steps.append(
                            "💡 Model's chain-of-thought:\n"
                            f"{thinking_text[:500]}"
                            f"{'...' if len(thinking_text) > 500 else ''}"
                        )
                    resp = ollama_response.message.content
                    timings["prompt_eval_count"] = getattr(
                        ollama_response, "prompt_eval_count", None
                    )
                    timings["eval_count"] = getattr(
                        ollama_response, "eval_count", None
                    )
                    _record_ollama_durations(timings, ollama_response)
                    return resp
                except Exception as e:
                    logger.warning(
                        f"Thinking mode failed, falling back to direct LLM: {e}"
                    )
            ai_message = llm.invoke(prepared.messages)
            meta = getattr(ai_message, "response_metadata", {}) or {}
            timings["prompt_eval_count"] = meta.get("prompt_eval_count")
            timings["eval_count"] = meta.get("eval_count")
            _record_ollama_durations(timings, meta)
            return ai_message.content

        def _call_groq() -> str:
            import httpx

            api_key = getattr(settings, "GROQ_API_KEY", None)
            if not api_key:
                raise RuntimeError("Groq API key not configured")
            groq_model = getattr(settings, "GROQ_MODEL", "openai/gpt-oss-20b")
            url = f"{getattr(settings, 'GROQ_API_BASE', 'https://api.groq.com/openai/v1').rstrip('/')}/chat/completions"
            resp = httpx.post(
                url,
                json={
                    "model": groq_model,
                    "messages": prepared.messages,
                    "stream": False,
                    "temperature": getattr(settings, "GROQ_TEMPERATURE", 0.1),
                    "max_tokens": getattr(settings, "GROQ_MAX_TOKENS", 512),
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=getattr(settings, "GROQ_TIMEOUT_SECONDS", 30.0),
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Groq API error {resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
            usage = data.get("usage") or {}
            timings["prompt_eval_count"] = usage.get("prompt_tokens")
            timings["eval_count"] = usage.get("completion_tokens")
            choices = data.get("choices") or []
            return choices[0]["message"]["content"] if choices else ""

        provider_used = primary_name
        fallback_occurred = False
        response = None

        try:
            if primary_name == "groq":
                response = _call_groq()
            else:
                response = _call_ollama()
        except Exception as primary_err:
            logger.warning(
                f"Primary sync generation ({primary_name}) failed: {primary_err}"
            )
            if fallback_name is not None:
                prepared.reasoning_steps.append(
                    f"⚠️ Primary ({primary_name}) failed: {primary_err}. Switching to {fallback_name} fallback."
                )
                try:
                    provider_used = fallback_name
                    fallback_occurred = True
                    if fallback_name == "groq":
                        response = _call_groq()
                    else:
                        response = _call_ollama()
                except Exception as fb_err:
                    if prepared.mode == "direct":
                        timings["generation_seconds"] = round(
                            time.perf_counter() - t_generation_start, 2
                        )
                        return "No PDFs found to query."
                    raise RuntimeError(
                        f"Both providers failed. Primary ({primary_name}): {primary_err}; Fallback ({fallback_name}): {fb_err}"
                    ) from fb_err
            else:
                if prepared.mode == "direct":
                    timings["generation_seconds"] = round(
                        time.perf_counter() - t_generation_start, 2
                    )
                    return "No PDFs found to query."
                raise

        timings["generation_seconds"] = round(
            time.perf_counter() - t_generation_start, 2
        )
        timings["provider"] = provider_used
        timings["mode"] = mode_val
        timings["fallback_occurred"] = fallback_occurred

        if prepared.mode == "direct":
            prepared.reasoning_steps.append(
                f"✨ Direct response generated using {provider_used} ({prepared.model})!"
            )
        else:
            prepared.reasoning_steps.append(
                f"✨ Answer generated successfully using {provider_used}!"
            )

        return str(response) if response is not None else ""

    def query_multi_pdf(
        self,
        question: str,
        model: str,
        pdf_ids: Optional[List[str]],
        db: Session,
        mode: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[str], Dict[str, Any]]:
        """Query across multiple PDFs with source attribution (non-streaming)."""
        t_start = time.perf_counter()
        prepared = self.prepare_query(question, model, pdf_ids, db)
        answer = self._generate_sync(prepared, mode=mode, provider=provider)
        prepared.timings["total_seconds"] = round(
            time.perf_counter() - t_start, 2
        )
        return answer, prepared.sources, prepared.reasoning_steps, prepared.timings

    async def stream_query(
        self,
        question: str,
        model: str,
        pdf_ids: Optional[List[str]],
        db: Session,
        provider: Optional[StreamingChatProvider] = None,
        mode: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a genuine token-by-token answer with provider fallback.

        Modes:
        - normal: Ollama primary, Groq fallback.
        - presentation: Groq primary, Ollama fallback.
        - offline: Ollama only (no external calls).
        """
        t_start = time.perf_counter()
        mode_val = (mode or getattr(settings, "RAG_MODE", "normal")).lower()
        if mode_val not in ("normal", "presentation", "offline"):
            raise ValueError(
                f"Invalid mode '{mode_val}'. Must be 'normal', 'presentation', or 'offline'."
            )

        prepared = await asyncio.to_thread(
            self.prepare_query, question, model, pdf_ids, db
        )

        # Resolve primary and fallback providers
        if provider is not None:
            # Caller supplied a specific provider — no automatic fallback.
            # Tests pass FakeProvider and control the stream fully.
            primary_provider = provider
            fallback_provider = None
        elif mode_val == "presentation":
            primary_provider = GroqStreamingProvider()
            fallback_provider = OllamaStreamingProvider()
        elif mode_val == "offline":
            primary_provider = OllamaStreamingProvider()
            fallback_provider = None
        else:  # normal
            primary_provider = OllamaStreamingProvider()
            fallback_provider = GroqStreamingProvider()

        yield StreamEvent(
            "meta",
            {
                "model": prepared.model,
                "provider": primary_provider.name,
                "mode": mode_val,
                "pdfs_queried": prepared.pdfs_queried,
                "use_thinking": prepared.use_thinking,
                "timings": prepared.timings,
            },
        )
        for step in prepared.reasoning_steps:
            yield StreamEvent("reasoning", {"text": step})
        if prepared.sources:
            yield StreamEvent("sources", {"sources": prepared.sources})

        t_generation_start = time.perf_counter()
        first_token_at: Optional[float] = None
        first_thinking_at: Optional[float] = None
        answer_parts: List[str] = []
        thinking_parts: List[str] = []
        final_metrics: Dict[str, Any] = {}
        completed = False
        provider_used = primary_provider.name
        fallback_occurred = False

        # First-token timeout for primary provider
        primary_timeout: Optional[float] = None
        if fallback_provider is not None:
            if mode_val == "normal":
                primary_timeout = getattr(settings, "OLLAMA_FALLBACK_TIMEOUT", 15.0)
            elif mode_val == "presentation":
                primary_timeout = getattr(settings, "GROQ_TIMEOUT_SECONDS", 30.0)

        async def _consume(target_p: StreamingChatProvider, timeout_first: Optional[float]):
            nonlocal first_token_at, first_thinking_at, final_metrics, completed
            gen = target_p.stream_chat(
                model=model,
                messages=prepared.messages,
                think=prepared.use_thinking if target_p.name == "ollama" else False,
            )
            async with aclosing(gen) as chunks:
                if timeout_first is not None and timeout_first > 0:
                    try:
                        first_chunk = await asyncio.wait_for(chunks.__anext__(), timeout=timeout_first)
                    except asyncio.TimeoutError:
                        raise ProviderTimeoutError(
                            f"{target_p.name.capitalize()} exceeded threshold ({timeout_first}s) before first token"
                        )
                    except StopAsyncIteration:
                        return

                    if first_chunk.thinking:
                        if first_thinking_at is None:
                            first_thinking_at = time.perf_counter()
                        thinking_parts.append(first_chunk.thinking)
                        yield StreamEvent("reasoning_delta", {"text": first_chunk.thinking})
                    if first_chunk.content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        answer_parts.append(first_chunk.content)
                        yield StreamEvent("token", {"text": first_chunk.content})
                    if first_chunk.done:
                        final_metrics = first_chunk.metrics or {}
                        completed = True
                        return

                async for chunk in chunks:
                    if chunk.thinking:
                        if first_thinking_at is None:
                            first_thinking_at = time.perf_counter()
                        thinking_parts.append(chunk.thinking)
                        yield StreamEvent("reasoning_delta", {"text": chunk.thinking})
                    if chunk.content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        answer_parts.append(chunk.content)
                        yield StreamEvent("token", {"text": chunk.content})
                    if chunk.done:
                        final_metrics = chunk.metrics or {}
                        completed = True
                        return

        primary_failed = False
        primary_error_msg = ""
        try:
            async with aclosing(_consume(primary_provider, primary_timeout)) as _primary_iter:
                async for ev in _primary_iter:
                    yield ev
        except (ProviderError, Exception) as e:
            logger.warning(f"Primary provider {primary_provider.name} failed: {e}")
            primary_failed = True
            primary_error_msg = str(e)

        if primary_failed:
            if fallback_provider is not None:
                fallback_occurred = True
                provider_used = fallback_provider.name
                yield StreamEvent(
                    "reasoning",
                    {
                        "text": (
                            f"⚡ Primary ({primary_provider.name}) failed or timed out "
                            f"({primary_error_msg}). Switched to {fallback_provider.name} fallback."
                        )
                    },
                )
                try:
                    async with aclosing(_consume(fallback_provider, None)) as _fb_iter:
                        async for ev in _fb_iter:
                            yield ev
                except Exception as fb_err:
                    logger.error(f"Fallback provider {fallback_provider.name} failed: {fb_err}")
                    yield StreamEvent(
                        "error",
                        {
                            "error": (
                                f"Both providers failed. Primary ({primary_provider.name}): "
                                f"{primary_error_msg}; Fallback ({fallback_provider.name}): {fb_err}"
                            )
                        },
                    )
                    return
            else:
                yield StreamEvent(
                    "error",
                    {
                        "error": f"Provider {primary_provider.name} failed in {mode_val} mode: {primary_error_msg}"
                    },
                )
                return

        if not completed:
            raise RuntimeError("Upstream stream ended before completion")

        timings = dict(prepared.timings)
        timings["provider"] = provider_used
        timings["mode"] = mode_val
        timings["fallback_occurred"] = fallback_occurred

        if provider_used == "groq":
            usage = final_metrics.get("usage") or {}
            timings["prompt_eval_count"] = usage.get("prompt_tokens")
            timings["eval_count"] = usage.get("completion_tokens")
            timings["total_tokens"] = usage.get("total_tokens")
        else:
            # Ollama and any test/fake provider — read metrics directly from the chunk dict
            for key in ("prompt_eval_count", "eval_count"):
                timings[key] = final_metrics.get(key)
            _record_ollama_durations(timings, final_metrics)

        now = time.perf_counter()
        timings["generation_seconds"] = round(now - t_generation_start, 2)
        timings["ttft_seconds"] = (
            round(first_token_at - t_generation_start, 2)
            if first_token_at is not None
            else None
        )
        timings["reasoning_ttft_seconds"] = (
            round(first_thinking_at - t_generation_start, 2)
            if first_thinking_at is not None
            else None
        )
        timings["total_seconds"] = round(now - t_start, 2)

        answer = "".join(answer_parts)
        if not answer.strip():
            yield StreamEvent(
                "error", {"error": "Model returned an empty response."}
            )
            return

        yield StreamEvent(
            "done",
            {
                "answer": answer,
                "sources": prepared.sources,
                "reasoning": "".join(thinking_parts),
                "timings": timings,
                "provider": provider_used,
                "mode": mode_val,
            },
        )

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]],
        db: Session,
    ) -> ChatMessage:
        """Save chat message to database.

        Args:
            session_id: Chat session identifier
            role: Message role (user or assistant)
            content: Message content
            sources: Source documents (for assistant messages)
            db: Database session

        Returns:
            Saved chat message
        """
        # Ensure session exists
        session = (
            db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
        if not session:
            session = ChatSession(
                session_id=session_id,
                created_at=datetime.now(),
                last_active=datetime.now(),
            )
            db.add(session)
        else:
            session.last_active = datetime.now()

        # Save message
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            timestamp=datetime.now(),
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    def get_session_messages(
        self, session_id: str, db: Session
    ) -> List[ChatMessage]:
        """Get all messages for a session.

        Args:
            session_id: Chat session identifier
            db: Database session

        Returns:
            List of chat messages
        """
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp)
            .all()
        )


def _role_value(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", "user")
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }.get(str(role), str(role))
