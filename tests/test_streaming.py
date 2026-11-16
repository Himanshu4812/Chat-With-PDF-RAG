"""Tests for genuine token streaming and save-on-completion semantics."""
import unittest
from unittest.mock import patch

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.database import Base, ChatMessage
from src.api.dependencies import get_db, get_rag_service
from src.api.main import app
from src.api.routers.query import _active_requests
from src.api.services.generation import ChatChunk
from src.api.services.rag_service import (
    PreparedQuery,
    RAGService,
    StreamEvent,
)


class FakeProvider:
    """Deterministic provider used instead of Ollama."""

    name = "fake"  # Required by stream_query provider routing

    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.received = None

    async def stream_chat(self, *, model, messages, think=False):
        self.received = {
            "model": model,
            "messages": messages,
            "think": think,
        }
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            self.closed = True


def _prepared(use_thinking=False):
    return PreparedQuery(
        mode="rag",
        messages=[{"role": "user", "content": "question"}],
        sources=[
            {
                "pdf_name": "sample.pdf",
                "pdf_id": "pdf-1",
                "chunk_index": 2,
                "page_number": 3,
            }
        ],
        reasoning_steps=["📚 Searching across 1 PDF(s)"],
        timings={"retrieval_seconds": 0.01},
        pdfs_queried=1,
        use_thinking=use_thinking,
        question="question",
        model="fake-model",
    )


class TestStreamQuery(unittest.IsolatedAsyncioTestCase):
    async def test_streams_tokens_incremental_and_records_ttft(self):
        service = RAGService()
        provider = FakeProvider(
            [
                ChatChunk(content="Hello"),
                ChatChunk(content=" world"),
                ChatChunk(
                    done=True,
                    metrics={
                        "prompt_eval_count": 5,
                        "eval_count": 2,
                        "prompt_eval_duration": 500_000_000,
                        "eval_duration": 1_000_000_000,
                    },
                ),
            ]
        )

        with patch.object(service, "prepare_query", return_value=_prepared()):
            events = [
                event
                async for event in service.stream_query(
                    "question", "fake-model", ["pdf-1"], db=None, provider=provider
                )
            ]

        names = [event.event for event in events]
        self.assertEqual(names[0], "meta")
        self.assertIn("reasoning", names)
        self.assertIn("sources", names)
        self.assertEqual(
            [e.data["text"] for e in events if e.event == "token"],
            ["Hello", " world"],
        )

        done = events[-1]
        self.assertEqual(done.event, "done")
        self.assertEqual(done.data["answer"], "Hello world")
        self.assertEqual(done.data["sources"][0]["page_number"], 3)
        self.assertIsNotNone(done.data["timings"]["ttft_seconds"])
        self.assertEqual(done.data["timings"]["eval_tokens_per_second"], 2.0)
        self.assertEqual(done.data["timings"]["eval_count"], 2)

        # The provider received the prepared messages, not a second prompt.
        self.assertEqual(
            provider.received["messages"], [{"role": "user", "content": "question"}]
        )

    async def test_client_cancellation_closes_upstream(self):
        service = RAGService()
        provider = FakeProvider(
            [ChatChunk(content="first"), ChatChunk(content="second")]
        )

        with patch.object(service, "prepare_query", return_value=_prepared()):
            stream = service.stream_query(
                "question", "fake-model", ["pdf-1"], db=None, provider=provider
            )
            first = await stream.__anext__()
            self.assertEqual(first.event, "meta")

            # Consume until the first token, then cancel like a Stop click.
            while True:
                event = await stream.__anext__()
                if event.event == "token":
                    break
            await stream.aclose()

        self.assertTrue(provider.closed)

    async def test_stream_without_done_raises_and_never_completes(self):
        service = RAGService()
        provider = FakeProvider([ChatChunk(content="partial")])

        with patch.object(service, "prepare_query", return_value=_prepared()):
            with self.assertRaises(RuntimeError):
                async for _ in service.stream_query(
                    "question", "fake-model", ["pdf-1"], db=None, provider=provider
                ):
                    pass

    async def test_empty_answer_emits_error_not_done(self):
        service = RAGService()
        provider = FakeProvider([ChatChunk(done=True, metrics={})])

        with patch.object(service, "prepare_query", return_value=_prepared()):
            events = [
                event
                async for event in service.stream_query(
                    "question", "fake-model", ["pdf-1"], db=None, provider=provider
                )
            ]

        self.assertEqual(events[-1].event, "error")
        self.assertNotIn("done", [event.event for event in events])

    def test_empty_pdf_list_is_direct_without_db_access(self):
        service = RAGService()

        class ExplodingDb:
            def query(self, *_args, **_kwargs):
                raise AssertionError("DB must not be queried when pdf_ids=[]")

        prepared = service.prepare_query(
            "hello", "phi4-mini:3.8b", [], ExplodingDb()
        )
        self.assertEqual(prepared.mode, "direct")
        self.assertEqual(prepared.sources, [])
        self.assertFalse(prepared.use_thinking)


class EndpointService(RAGService):
    """Service double that lets endpoint tests control the SSE sequence."""

    def __init__(self, events):
        super().__init__()
        self.events = events

    async def stream_query(self, **_kwargs):
        for event in self.events:
            yield event


class TestStreamingEndpoint(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        def override_get_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(
            transport=self.transport, base_url="http://test"
        )
        _active_requests.clear()

    async def asyncTearDown(self):
        await self.client.aclose()
        app.dependency_overrides.clear()
        Base.metadata.drop_all(self.engine)
        _active_requests.clear()

    def _use_service(self, events):
        app.dependency_overrides[get_rag_service] = lambda: EndpointService(events)

    async def _post(self, request_id="req-1"):
        response = await self.client.post(
            "/api/v1/query/stream",
            json={
                "question": "What is this about?",
                "model": "fake-model",
                "pdf_ids": [],
                "session_id": "session-1",
                "request_id": request_id,
            },
        )
        body = "".join([chunk async for chunk in response.aiter_text()])
        return response, body

    def _messages(self):
        db = self.Session()
        try:
            return db.query(ChatMessage).order_by(ChatMessage.message_id).all()
        finally:
            db.close()

    async def test_completed_stream_saves_user_and_assistant(self):
        self._use_service(
            [
                StreamEvent("meta", {"timings": {}, "pdfs_queried": 0}),
                StreamEvent("token", {"text": "Hello"}),
                StreamEvent(
                    "done",
                    {
                        "answer": "Hello",
                        "sources": [],
                        "reasoning": "",
                        "timings": {"ttft_seconds": 0.1, "total_seconds": 0.2},
                    },
                ),
            ]
        )

        response, body = await self._post()

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", body)
        messages = self._messages()
        self.assertEqual([m.role for m in messages], ["user", "assistant"])
        self.assertEqual(messages[1].content, "Hello")

    async def test_incomplete_stream_saves_no_assistant(self):
        self._use_service(
            [
                StreamEvent("meta", {"timings": {}, "pdfs_queried": 0}),
                StreamEvent("token", {"text": "partial"}),
            ]
        )

        _, body = await self._post()

        self.assertIn("event: error", body)
        self.assertNotIn("event: done", body)
        messages = self._messages()
        self.assertEqual([m.role for m in messages], ["user"])

    async def test_provider_error_is_not_followed_by_generic_error(self):
        self._use_service(
            [
                StreamEvent("meta", {"timings": {}, "pdfs_queried": 0}),
                StreamEvent(
                    "error",
                    {"error": "Both providers failed. Primary (ollama): timeout"},
                ),
            ]
        )

        _, body = await self._post()

        self.assertEqual(body.count("event: error"), 1)
        self.assertIn("Both providers failed", body)
        self.assertNotIn("ended before the answer was completed", body)
        messages = self._messages()
        self.assertEqual([m.role for m in messages], ["user"])

    async def test_duplicate_request_id_is_rejected(self):
        self._use_service(
            [
                StreamEvent(
                    "done",
                    {
                        "answer": "ok",
                        "sources": [],
                        "reasoning": "",
                        "timings": {},
                    },
                )
            ]
        )
        _active_requests.add("busy-request")

        response, _ = await self._post(request_id="busy-request")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self._messages(), [])
