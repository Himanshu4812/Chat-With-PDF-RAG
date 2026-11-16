"""Stage 4 — Groq Cloud integration tests.

Ten required scenarios plus TTFT/latency measurement helpers.
All heavy I/O (Ollama, Groq HTTP) is mocked; only the logic under test is exercised.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests\\test_stage4_groq.py -v
"""
from __future__ import annotations

import asyncio
import time
import unittest
from typing import AsyncIterator, Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.services.generation import (
    ChatChunk,
    GroqStreamingProvider,
    OllamaStreamingProvider,
    ProviderAuthError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.api.services.rag_service import PreparedQuery, RAGService, StreamEvent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prepared(
    sources: Optional[List[Dict]] = None,
    model: str = "phi4-mini:3.8b",
) -> PreparedQuery:
    """Return a minimal PreparedQuery for unit tests."""
    return PreparedQuery(
        mode="rag",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        sources=sources
        or [
            {
                "pdf_name": "sample.pdf",
                "pdf_id": "pdf-1",
                "chunk_index": 0,
                "page_number": 1,
                "text": "France is a country in Western Europe. Its capital is Paris.",
            }
        ],
        reasoning_steps=["📚 Searching across 1 PDF(s)"],
        timings={"retrieval_seconds": 0.01},
        pdfs_queried=1,
        use_thinking=False,
        question="What is the capital of France?",
        model=model,
    )


async def _collect(aiter) -> List[StreamEvent]:
    """Drain an async generator into a list."""
    return [ev async for ev in aiter]


def _tokens(events: List[StreamEvent]) -> str:
    return "".join(e.data["text"] for e in events if e.event == "token")


def _event_names(events: List[StreamEvent]) -> List[str]:
    return [e.event for e in events]


# ---------------------------------------------------------------------------
# Fake provider implementations
# ---------------------------------------------------------------------------


class FakeOllamaProvider:
    """Fake Ollama provider that streams successfully."""

    name = "ollama"

    def __init__(self, chunks: Optional[List[ChatChunk]] = None):
        self.chunks = chunks or [
            ChatChunk(content="Paris"),
            ChatChunk(content=" is the capital."),
            ChatChunk(
                done=True,
                metrics={
                    "prompt_eval_count": 10,
                    "eval_count": 6,
                    "prompt_eval_duration": 500_000_000,
                    "eval_duration": 1_000_000_000,
                },
            ),
        ]
        self.called = False
        self.closed = False

    async def stream_chat(self, *, model, messages, think=False) -> AsyncIterator[ChatChunk]:
        self.called = True
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            self.closed = True


class FakeGroqProvider:
    """Fake Groq provider that streams successfully."""

    name = "groq"

    def __init__(self, chunks: Optional[List[ChatChunk]] = None):
        self.chunks = chunks or [
            ChatChunk(content="Paris", metrics={"usage": {}, "model": "llama-3.3-70b-versatile", "provider": "groq"}),
            ChatChunk(content=" is the capital.", metrics={"usage": {}, "model": "llama-3.3-70b-versatile", "provider": "groq"}),
            ChatChunk(
                done=True,
                metrics={
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                    "model": "llama-3.3-70b-versatile",
                    "provider": "groq",
                },
            ),
        ]
        self.called = False
        self.closed = False

    async def stream_chat(self, *, model, messages, think=False) -> AsyncIterator[ChatChunk]:
        self.called = True
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            self.closed = True


class SlowOllamaProvider:
    """Simulates Ollama exceeding the first-token timeout."""

    name = "ollama"

    def __init__(self, delay: float = 20.0):
        self.delay = delay
        self.closed = False

    async def stream_chat(self, *, model, messages, think=False) -> AsyncIterator[ChatChunk]:
        try:
            await asyncio.sleep(self.delay)
            yield ChatChunk(content="too late")
        finally:
            self.closed = True


class FailingGroqProvider:
    """Groq provider that raises a configurable error."""

    name = "groq"

    def __init__(self, error: Exception):
        self.error = error
        self.called = False

    async def stream_chat(self, *, model, messages, think=False) -> AsyncIterator[ChatChunk]:
        self.called = True
        raise self.error
        # Make it an async generator
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# SCENARIO 1 — Normal mode: Ollama responds successfully
# ---------------------------------------------------------------------------


class TestScenario1_OllamaSuccess(unittest.IsolatedAsyncioTestCase):
    """Normal mode — Ollama primary succeeds; Groq is never called."""

    async def test_ollama_primary_success(self):
        service = RAGService()
        ollama_p = FakeOllamaProvider()
        groq_p = FakeGroqProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            events = await _collect(
                service.stream_query(
                    "What is the capital of France?",
                    "phi4-mini:3.8b",
                    ["pdf-1"],
                    db=None,
                    provider=ollama_p,
                    mode="normal",
                )
            )

        names = _event_names(events)
        self.assertIn("meta", names)
        self.assertIn("token", names)
        self.assertIn("done", names)
        self.assertNotIn("error", names)

        # Answer assembled correctly
        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["answer"], "Paris is the capital.")

        # Groq never called
        self.assertFalse(groq_p.called)

        # Provider recorded in timings
        self.assertEqual(done.data["provider"], "ollama")
        self.assertEqual(done.data["mode"], "normal")

    async def test_sources_preserved_in_done_event(self):
        """PDF sources are present in done event (citation preservation)."""
        service = RAGService()
        ollama_p = FakeOllamaProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            events = await _collect(
                service.stream_query(
                    "question", "phi4-mini:3.8b", ["pdf-1"], db=None, provider=ollama_p
                )
            )

        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(len(done.data["sources"]), 1)
        self.assertEqual(done.data["sources"][0]["pdf_name"], "sample.pdf")
        self.assertEqual(done.data["sources"][0]["page_number"], 1)


# ---------------------------------------------------------------------------
# SCENARIO 2 — Normal mode: Ollama exceeds threshold → Groq fallback
# ---------------------------------------------------------------------------


class TestScenario2_OllamaTimeoutGroqFallback(unittest.IsolatedAsyncioTestCase):
    """Normal mode — Ollama times out before first token; Groq takes over."""

    async def test_fallback_to_groq_on_timeout(self):
        service = RAGService()

        slow_ollama = SlowOllamaProvider(delay=20.0)
        groq_p = FakeGroqProvider()

        prepared = _prepared()

        with patch.object(service, "prepare_query", return_value=prepared):
            # Keep the timeout independent of the configured default.
            with patch(
                "src.api.services.rag_service.settings.OLLAMA_FALLBACK_TIMEOUT",
                0.1,
            ):
                # Patch the providers to inject our fakes
                with patch(
                    "src.api.services.rag_service.OllamaStreamingProvider",
                    return_value=slow_ollama,
                ):
                    with patch(
                        "src.api.services.rag_service.GroqStreamingProvider",
                        return_value=groq_p,
                    ):
                        events = await _collect(
                            service.stream_query(
                                "question",
                                "phi4-mini:3.8b",
                                ["pdf-1"],
                                db=None,
                                mode="normal",
                            )
                        )

        names = _event_names(events)
        self.assertIn("token", names, "Fallback should produce tokens")
        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["provider"], "groq")
        self.assertTrue(done.data["timings"]["fallback_occurred"])

        # A reasoning event must explain the switch
        reasoning_texts = [e.data.get("text", "") for e in events if e.event == "reasoning"]
        self.assertTrue(
            any("fallback" in t.lower() or "switched" in t.lower() or "timed out" in t.lower()
                for t in reasoning_texts),
            f"Expected fallback reasoning event; got: {reasoning_texts}",
        )


# ---------------------------------------------------------------------------
# SCENARIO 3 — Presentation mode: Groq is primary, Ollama fallback
# ---------------------------------------------------------------------------


class TestScenario3_PresentationMode(unittest.IsolatedAsyncioTestCase):
    """Presentation mode — Groq is primary; Ollama is not called."""

    async def test_groq_is_primary_in_presentation_mode(self):
        service = RAGService()

        groq_p = FakeGroqProvider()
        ollama_p = FakeOllamaProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with patch(
                "src.api.services.rag_service.GroqStreamingProvider",
                return_value=groq_p,
            ):
                with patch(
                    "src.api.services.rag_service.OllamaStreamingProvider",
                    return_value=ollama_p,
                ):
                    events = await _collect(
                        service.stream_query(
                            "question",
                            "phi4-mini:3.8b",
                            ["pdf-1"],
                            db=None,
                            mode="presentation",
                        )
                    )

        names = _event_names(events)
        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["provider"], "groq")
        self.assertEqual(done.data["mode"], "presentation")

        # Ollama must not have been called
        self.assertFalse(ollama_p.called)

        # meta event must announce groq as primary
        meta = next(e for e in events if e.event == "meta")
        self.assertEqual(meta.data["provider"], "groq")


# ---------------------------------------------------------------------------
# SCENARIO 4 — Offline mode: Ollama only, no external calls
# ---------------------------------------------------------------------------


class TestScenario4_OfflineMode(unittest.IsolatedAsyncioTestCase):
    """Offline mode — only Ollama is used; Groq is never instantiated."""

    async def test_offline_mode_no_groq_call(self):
        service = RAGService()
        ollama_p = FakeOllamaProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with patch(
                "src.api.services.rag_service.OllamaStreamingProvider",
                return_value=ollama_p,
            ) as mock_ollama_cls:
                with patch(
                    "src.api.services.rag_service.GroqStreamingProvider"
                ) as mock_groq_cls:
                    events = await _collect(
                        service.stream_query(
                            "question",
                            "phi4-mini:3.8b",
                            ["pdf-1"],
                            db=None,
                            mode="offline",
                        )
                    )

        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["provider"], "ollama")
        self.assertEqual(done.data["mode"], "offline")

        # GroqStreamingProvider must NOT be instantiated
        mock_groq_cls.assert_not_called()

    async def test_offline_mode_ollama_failure_returns_error_not_groq(self):
        """If Ollama fails in offline mode, return error — no Groq fallback."""
        service = RAGService()
        failing_ollama = FailingGroqProvider(ProviderNetworkError("Ollama unreachable"))
        failing_ollama.name = "ollama"  # pose as ollama

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with patch(
                "src.api.services.rag_service.OllamaStreamingProvider",
                return_value=failing_ollama,
            ):
                with patch(
                    "src.api.services.rag_service.GroqStreamingProvider"
                ) as mock_groq_cls:
                    events = await _collect(
                        service.stream_query(
                            "question",
                            "phi4-mini:3.8b",
                            ["pdf-1"],
                            db=None,
                            mode="offline",
                        )
                    )

        # Must return an error, not groq
        names = _event_names(events)
        self.assertIn("error", names)
        self.assertNotIn("done", names)
        mock_groq_cls.assert_not_called()


# ---------------------------------------------------------------------------
# SCENARIO 5 — Groq HTTP 429 handling
# ---------------------------------------------------------------------------


class TestScenario5_Groq429(unittest.IsolatedAsyncioTestCase):
    """Groq rate-limit (HTTP 429) is surfaced as ProviderRateLimitError."""

    async def test_groq_429_propagates_as_rate_limit_error(self):
        """Presentation mode: Groq 429 → Ollama fallback."""
        service = RAGService()

        groq_rate_limited = FailingGroqProvider(
            ProviderRateLimitError("Groq rate limit exceeded (HTTP 429). Free-tier limit reached.")
        )
        ollama_p = FakeOllamaProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with patch(
                "src.api.services.rag_service.GroqStreamingProvider",
                return_value=groq_rate_limited,
            ):
                with patch(
                    "src.api.services.rag_service.OllamaStreamingProvider",
                    return_value=ollama_p,
                ):
                    events = await _collect(
                        service.stream_query(
                            "question",
                            "phi4-mini:3.8b",
                            ["pdf-1"],
                            db=None,
                            mode="presentation",
                        )
                    )

        # Should fall back to Ollama
        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["provider"], "ollama")
        self.assertTrue(done.data["timings"]["fallback_occurred"])

        # Reasoning must mention the switch
        reasoning_texts = [e.data.get("text", "") for e in events if e.event == "reasoning"]
        self.assertTrue(any("429" in t or "rate limit" in t.lower() for t in reasoning_texts))

    async def test_groq_rate_limit_error_class(self):
        """GroqStreamingProvider raises ProviderRateLimitError on 429."""
        provider = GroqStreamingProvider(api_key="gsk_fake_key_for_test")

        # Build a fake 429 response as a sync MagicMock (httpx response is not awaited)
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}

        # client.stream() must be a sync context manager returning the mock_response
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_ctx

        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        client_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client_ctx):
            with self.assertRaises(ProviderRateLimitError):
                async for _ in provider.stream_chat(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "hi"}],
                ):
                    pass


# ---------------------------------------------------------------------------
# SCENARIO 6 — Invalid API key (HTTP 401)
# ---------------------------------------------------------------------------


class TestScenario6_InvalidApiKey(unittest.IsolatedAsyncioTestCase):
    """Groq raises ProviderAuthError on HTTP 401 / missing key."""

    async def test_missing_key_raises_auth_error(self):
        """GroqStreamingProvider with no key raises ProviderAuthError immediately."""
        provider = GroqStreamingProvider(api_key=None)
        # Temporarily ensure settings has no key
        with patch("src.api.services.generation.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = None
            mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
            mock_settings.GROQ_API_BASE = "https://api.groq.com/openai/v1"
            mock_settings.GROQ_TIMEOUT_SECONDS = 30.0
            mock_settings.GROQ_TEMPERATURE = 0.1
            mock_settings.GROQ_MAX_TOKENS = 512

            p = GroqStreamingProvider(api_key=None)
            p.api_key = None  # explicit

            with self.assertRaises(ProviderAuthError):
                async for _ in p.stream_chat(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "test"}],
                ):
                    pass

    async def test_invalid_key_401_raises_auth_error(self):
        """HTTP 401 response from Groq raises ProviderAuthError."""
        provider = GroqStreamingProvider(api_key="gsk_invalid_key_for_test")

        mock_response = MagicMock()
        mock_response.status_code = 401

        # client.stream() must be a sync context manager returning the mock_response
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_ctx

        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        client_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client_ctx):
            with self.assertRaises(ProviderAuthError):
                async for _ in provider.stream_chat(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "test"}],
                ):
                    pass

    async def test_invalid_key_falls_back_to_ollama_in_presentation_mode(self):
        """In presentation mode, Groq auth failure falls back to Ollama."""
        service = RAGService()

        auth_fail_groq = FailingGroqProvider(
            ProviderAuthError("Invalid or unauthorized Groq API key (HTTP 401)")
        )
        ollama_p = FakeOllamaProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with patch(
                "src.api.services.rag_service.GroqStreamingProvider",
                return_value=auth_fail_groq,
            ):
                with patch(
                    "src.api.services.rag_service.OllamaStreamingProvider",
                    return_value=ollama_p,
                ):
                    events = await _collect(
                        service.stream_query(
                            "question",
                            "phi4-mini:3.8b",
                            ["pdf-1"],
                            db=None,
                            mode="presentation",
                        )
                    )

        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["provider"], "ollama")


# ---------------------------------------------------------------------------
# SCENARIO 7 — Internet disconnection / network failure
# ---------------------------------------------------------------------------


class TestScenario7_NetworkFailure(unittest.IsolatedAsyncioTestCase):
    """ProviderNetworkError when Groq is unreachable."""

    async def test_groq_network_error_falls_back_to_ollama(self):
        service = RAGService()

        net_fail_groq = FailingGroqProvider(
            ProviderNetworkError("Failed to connect to Groq API: [WinError 10013] ...")
        )
        ollama_p = FakeOllamaProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with patch(
                "src.api.services.rag_service.GroqStreamingProvider",
                return_value=net_fail_groq,
            ):
                with patch(
                    "src.api.services.rag_service.OllamaStreamingProvider",
                    return_value=ollama_p,
                ):
                    events = await _collect(
                        service.stream_query(
                            "question",
                            "phi4-mini:3.8b",
                            ["pdf-1"],
                            db=None,
                            mode="presentation",
                        )
                    )

        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["provider"], "ollama")
        self.assertTrue(done.data["timings"]["fallback_occurred"])

    async def test_both_providers_fail_returns_error_not_fabricated(self):
        """If both providers fail, return error — never fabricate an answer."""
        service = RAGService()

        fail_groq = FailingGroqProvider(ProviderNetworkError("No internet"))
        fail_ollama = FailingGroqProvider(ProviderNetworkError("Ollama unreachable"))
        fail_ollama.name = "ollama"

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with patch(
                "src.api.services.rag_service.GroqStreamingProvider",
                return_value=fail_groq,
            ):
                with patch(
                    "src.api.services.rag_service.OllamaStreamingProvider",
                    return_value=fail_ollama,
                ):
                    events = await _collect(
                        service.stream_query(
                            "question",
                            "phi4-mini:3.8b",
                            ["pdf-1"],
                            db=None,
                            mode="presentation",
                        )
                    )

        names = _event_names(events)
        self.assertIn("error", names)
        self.assertNotIn("done", names)

        err = next(e for e in events if e.event == "error")
        self.assertIn("Both providers failed", err.data["error"])


# ---------------------------------------------------------------------------
# SCENARIO 8 — Cancellation during local generation
# ---------------------------------------------------------------------------


class TestScenario8_Cancellation(unittest.IsolatedAsyncioTestCase):
    """aclose() on the outer stream closes the upstream provider's connection."""

    async def test_cancel_during_ollama_generation_closes_upstream(self):
        service = RAGService()

        async def _infinite_provider(*, model, messages, think=False):
            try:
                i = 0
                while True:
                    yield ChatChunk(content=f"word{i} ")
                    i += 1
                    await asyncio.sleep(0)
            finally:
                infinite.closed = True

        class InfiniteOllama:
            name = "ollama"
            closed = False

            def stream_chat(self, *, model, messages, think=False):
                return _infinite_provider(model=model, messages=messages, think=think)

        infinite = InfiniteOllama()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            gen = service.stream_query(
                "question", "phi4-mini:3.8b", ["pdf-1"], db=None, provider=infinite
            )

            # Advance past meta, reasoning, sources
            tokens_seen = 0
            async for ev in gen:
                if ev.event == "token":
                    tokens_seen += 1
                    if tokens_seen >= 2:
                        break

            # Cancel — simulates the Stop button
            await gen.aclose()

        self.assertTrue(infinite.closed, "Provider stream must be closed on cancellation")


# ---------------------------------------------------------------------------
# SCENARIO 9 — Streaming from Groq (token-by-token SSE delivery)
# ---------------------------------------------------------------------------


class TestScenario9_GroqStreaming(unittest.IsolatedAsyncioTestCase):
    """Groq provider delivers genuine token-by-token streaming."""

    async def test_groq_tokens_arrive_incrementally(self):
        service = RAGService()

        groq_p = FakeGroqProvider(
            chunks=[
                ChatChunk(content="The", metrics={"usage": {}, "model": "llama-3.3-70b-versatile", "provider": "groq"}),
                ChatChunk(content=" capital", metrics={"usage": {}, "model": "llama-3.3-70b-versatile", "provider": "groq"}),
                ChatChunk(content=" is", metrics={"usage": {}, "model": "llama-3.3-70b-versatile", "provider": "groq"}),
                ChatChunk(content=" Paris.", metrics={"usage": {}, "model": "llama-3.3-70b-versatile", "provider": "groq"}),
                ChatChunk(
                    done=True,
                    metrics={
                        "usage": {"prompt_tokens": 15, "completion_tokens": 4, "total_tokens": 19},
                        "model": "llama-3.3-70b-versatile",
                        "provider": "groq",
                    },
                ),
            ]
        )

        with patch.object(service, "prepare_query", return_value=_prepared()):
            events = await _collect(
                service.stream_query(
                    "question",
                    "phi4-mini:3.8b",
                    ["pdf-1"],
                    db=None,
                    provider=groq_p,
                    mode="presentation",
                )
            )

        token_events = [e for e in events if e.event == "token"]
        # Each word must arrive as a separate token event (genuine streaming)
        self.assertGreaterEqual(len(token_events), 2, "Should have multiple token events")
        assembled = "".join(e.data["text"] for e in token_events)
        self.assertEqual(assembled, "The capital is Paris.")

        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["answer"], "The capital is Paris.")
        self.assertEqual(done.data["provider"], "groq")

    async def test_groq_ttft_recorded(self):
        """TTFT is recorded in timings for Groq provider."""
        service = RAGService()
        groq_p = FakeGroqProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            events = await _collect(
                service.stream_query(
                    "question",
                    "phi4-mini:3.8b",
                    ["pdf-1"],
                    db=None,
                    provider=groq_p,
                    mode="presentation",
                )
            )

        done = events[-1]
        self.assertIsNotNone(done.data["timings"].get("ttft_seconds"))
        self.assertGreaterEqual(done.data["timings"]["ttft_seconds"], 0)

    async def test_groq_usage_tokens_recorded(self):
        """Groq token usage is captured in timings."""
        service = RAGService()
        groq_p = FakeGroqProvider()

        with patch.object(service, "prepare_query", return_value=_prepared()):
            events = await _collect(
                service.stream_query(
                    "question",
                    "phi4-mini:3.8b",
                    ["pdf-1"],
                    db=None,
                    provider=groq_p,
                    mode="presentation",
                )
            )

        done = events[-1]
        timings = done.data["timings"]
        self.assertEqual(timings.get("prompt_eval_count"), 20)
        self.assertEqual(timings.get("eval_count"), 8)
        self.assertEqual(timings.get("total_tokens"), 28)


# ---------------------------------------------------------------------------
# SCENARIO 10 — PDF citations remain correct across providers
# ---------------------------------------------------------------------------


class TestScenario10_CitationPreservation(unittest.IsolatedAsyncioTestCase):
    """Sources/citations are preserved regardless of which provider answers."""

    SOURCES = [
        {
            "pdf_name": "research_paper.pdf",
            "pdf_id": "pdf-research",
            "chunk_index": 7,
            "page_number": 42,
            "text": "The model achieved 95% accuracy on the test set.",
        },
        {
            "pdf_name": "appendix.pdf",
            "pdf_id": "pdf-appendix",
            "chunk_index": 2,
            "page_number": 101,
            "text": "See appendix B for detailed methodology.",
        },
    ]

    async def _run_with_provider(self, provider, mode: str) -> List[StreamEvent]:
        service = RAGService()
        with patch.object(service, "prepare_query", return_value=_prepared(sources=self.SOURCES)):
            return await _collect(
                service.stream_query(
                    "question", "phi4-mini:3.8b", ["pdf-1"], db=None, provider=provider, mode=mode
                )
            )

    async def test_sources_preserved_via_ollama(self):
        events = await self._run_with_provider(FakeOllamaProvider(), "normal")
        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(len(done.data["sources"]), 2)
        self.assertEqual(done.data["sources"][0]["pdf_name"], "research_paper.pdf")
        self.assertEqual(done.data["sources"][0]["page_number"], 42)
        self.assertEqual(done.data["sources"][1]["page_number"], 101)

    async def test_sources_preserved_via_groq(self):
        events = await self._run_with_provider(FakeGroqProvider(), "presentation")
        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(len(done.data["sources"]), 2)
        self.assertEqual(done.data["sources"][0]["pdf_name"], "research_paper.pdf")
        self.assertEqual(done.data["sources"][0]["page_number"], 42)
        self.assertEqual(done.data["sources"][1]["chunk_index"], 2)

    async def test_sources_emitted_as_sources_event(self):
        """A 'sources' event is always emitted before generation starts."""
        events = await self._run_with_provider(FakeOllamaProvider(), "normal")
        source_events = [e for e in events if e.event == "sources"]
        self.assertEqual(len(source_events), 1)
        self.assertEqual(len(source_events[0].data["sources"]), 2)


# ---------------------------------------------------------------------------
# TTFT / Latency measurement helper (not a strict assertion)
# ---------------------------------------------------------------------------


class TestTTFTAndLatencyMeasurement(unittest.IsolatedAsyncioTestCase):
    """Measure TTFT and total time; recorded in done.timings."""

    async def _measure(self, provider, mode: str):
        service = RAGService()
        t0 = time.perf_counter()
        events = []
        first_token_wall: Optional[float] = None

        with patch.object(service, "prepare_query", return_value=_prepared()):
            async for ev in service.stream_query(
                "question", "phi4-mini:3.8b", ["pdf-1"], db=None, provider=provider, mode=mode
            ):
                events.append(ev)
                if ev.event == "token" and first_token_wall is None:
                    first_token_wall = time.perf_counter() - t0

        total_wall = time.perf_counter() - t0
        done = events[-1]
        internal_ttft = done.data["timings"].get("ttft_seconds")
        internal_total = done.data["timings"].get("total_seconds")
        return {
            "ttft_wall": round(first_token_wall or 0.0, 4),
            "total_wall": round(total_wall, 4),
            "ttft_internal": internal_ttft,
            "total_internal": internal_total,
            "provider": done.data.get("provider"),
        }

    async def test_ollama_ttft_recorded(self):
        result = await self._measure(FakeOllamaProvider(), "normal")
        self.assertIsNotNone(result["ttft_internal"], "TTFT should be recorded for Ollama")
        self.assertGreaterEqual(result["ttft_internal"], 0)
        self.assertGreaterEqual(result["total_internal"], result["ttft_internal"])

    async def test_groq_ttft_recorded(self):
        result = await self._measure(FakeGroqProvider(), "presentation")
        self.assertIsNotNone(result["ttft_internal"], "TTFT should be recorded for Groq")
        self.assertGreaterEqual(result["ttft_internal"], 0)
        self.assertGreaterEqual(result["total_internal"], result["ttft_internal"])

    async def test_latency_fields_present(self):
        """Both providers emit generation_seconds and total_seconds in done."""
        for provider, mode in [(FakeOllamaProvider(), "normal"), (FakeGroqProvider(), "presentation")]:
            result = await self._measure(provider, mode)
            self.assertIsNotNone(result["total_internal"])
            self.assertIsNotNone(result["ttft_internal"])


# ---------------------------------------------------------------------------
# Provider configuration tests
# ---------------------------------------------------------------------------


class TestProviderConfiguration(unittest.IsolatedAsyncioTestCase):
    """API key is never exposed; model is configurable."""

    def test_groq_provider_reads_api_key_from_settings(self):
        """GroqStreamingProvider reads key from settings, never from frontend."""
        with patch("src.api.services.generation.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "gsk_test_key"
            mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
            mock_settings.GROQ_API_BASE = "https://api.groq.com/openai/v1"
            mock_settings.GROQ_TIMEOUT_SECONDS = 30.0

            p = GroqStreamingProvider()
            self.assertEqual(p.api_key, "gsk_test_key")

    def test_groq_model_is_configurable(self):
        with patch("src.api.services.generation.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "gsk_test_key"
            mock_settings.GROQ_MODEL = "llama-3.1-8b-instant"
            mock_settings.GROQ_API_BASE = "https://api.groq.com/openai/v1"
            mock_settings.GROQ_TIMEOUT_SECONDS = 30.0

            p = GroqStreamingProvider()
            self.assertEqual(p.default_model, "llama-3.1-8b-instant")

    def test_ollama_model_name_resolves_to_groq_model(self):
        """Passing an Ollama model name to Groq resolves to the configured Groq model."""
        with patch("src.api.services.generation.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "gsk_test_key"
            mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
            mock_settings.GROQ_API_BASE = "https://api.groq.com/openai/v1"
            mock_settings.GROQ_TIMEOUT_SECONDS = 30.0

            p = GroqStreamingProvider()
            # Ollama-style name (has colon) → resolves to Groq model
            self.assertEqual(p._resolve_model("phi4-mini:3.8b"), "llama-3.3-70b-versatile")
            # Already a Groq model name → passes through
            self.assertEqual(
                p._resolve_model("llama-3.3-70b-versatile"), "llama-3.3-70b-versatile"
            )

    def test_api_key_not_in_resolved_model_name(self):
        """The API key must never appear in a model name or public attribute."""
        with patch("src.api.services.generation.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "gsk_super_secret_key_12345"
            mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
            mock_settings.GROQ_API_BASE = "https://api.groq.com/openai/v1"
            mock_settings.GROQ_TIMEOUT_SECONDS = 30.0

            p = GroqStreamingProvider()
            model = p._resolve_model("phi4-mini:3.8b")
            self.assertNotIn("gsk_super_secret_key_12345", model)
            self.assertNotIn("gsk_super_secret_key_12345", p.base_url)
            self.assertNotIn("gsk_super_secret_key_12345", p.default_model)
