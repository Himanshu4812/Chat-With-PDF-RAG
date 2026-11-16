"""RAG query endpoints."""
import asyncio
import json
import logging
import threading
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..dependencies import get_db, get_rag_service
from ..models import QueryRequest, QueryResponse, SourceInfo
from ..services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])

# Guard against duplicate generation for the same request id. Entries are
# released when the stream finishes, fails or is cancelled.
_active_requests: set[str] = set()
_active_requests_lock = threading.Lock()


def _acquire_request(request_id: str) -> bool:
    with _active_requests_lock:
        if request_id in _active_requests:
            return False
        _active_requests.add(request_id)
        return True


def _release_request(request_id: str) -> None:
    with _active_requests_lock:
        _active_requests.discard(request_id)


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/query", response_model=QueryResponse)
def query_pdfs(
    request: QueryRequest,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Query across PDFs with source attribution."""
    logger.info(
        f"📥 Received query request: question='{request.question[:50]}...', "
        f"model={request.model}"
    )

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"🔑 Session ID: {session_id}")

    # Save user message
    rag_service.save_message(
        session_id=session_id,
        role="user",
        content=request.question,
        sources=None,
        db=db,
    )
    logger.info("💾 User message saved")

    # Validate mode if specified
    if request.mode and request.mode.lower() not in ("normal", "presentation", "offline"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Must be 'normal', 'presentation', or 'offline'.",
        )

    # Query RAG
    logger.info("🚀 Starting RAG query...")
    try:
        answer, sources, reasoning_steps, timings = (
            rag_service.query_multi_pdf(
                question=request.question,
                model=request.model,
                pdf_ids=request.pdf_ids,
                db=db,
                mode=request.mode,
                provider=request.provider,
            )
        )
        logger.info(
            f"✅ RAG query complete: answer_length={len(answer)}, "
            f"sources_count={len(sources)}, timings={timings}"
        )
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() and "404" in error_msg:
            logger.error(f"❌ Model not found: {request.model}")
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model '{request.model}' not found. Please select a "
                    "different model from the dropdown or install it with: "
                    f"ollama pull {request.model}"
                ),
            )
        logger.error(f"❌ Query failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Query failed: {error_msg}")

    # Save assistant message
    message = rag_service.save_message(
        session_id=session_id,
        role="assistant",
        content=answer,
        sources=sources,
        db=db,
    )
    logger.info(f"💾 Assistant message saved with ID: {message.message_id}")

    response = QueryResponse(
        answer=answer,
        sources=[SourceInfo(**s) for s in sources],
        metadata={
            "model_used": request.model,
            "chunks_retrieved": len(sources),
            "pdfs_queried": len(set(s["pdf_id"] for s in sources)),
            "reasoning_steps": reasoning_steps,
            "timings": timings,
        },
        session_id=session_id,
        message_id=message.message_id,
    )

    logger.info(
        f"📤 Returning response: answer_length={len(response.answer)}, "
        f"sources={len(response.sources)}"
    )

    return response


@router.post("/query/stream")
async def query_pdfs_stream(
    request: QueryRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Stream a genuine token-by-token answer over Server-Sent Events.

    Events: meta, reasoning, reasoning_delta, sources, token, done, error.
    The assistant message is saved only after a completed (done) stream.
    """
    session_id = request.session_id or str(uuid.uuid4())
    request_key: Optional[str] = request.request_id
    if request_key and not _acquire_request(request_key):
        raise HTTPException(
            status_code=409,
            detail="A generation for this request is already in progress.",
        )

    if request.mode and request.mode.lower() not in ("normal", "presentation", "offline"):
        if request_key:
            _release_request(request_key)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Must be 'normal', 'presentation', or 'offline'.",
        )

    logger.info(
        f"📥 Streaming query: model={request.model}, mode={request.mode or 'default'}, "
        f"pdfs={'all' if request.pdf_ids is None else len(request.pdf_ids)}, "
        f"session={session_id}"
    )

    try:
        rag_service.save_message(
            session_id=session_id,
            role="user",
            content=request.question,
            sources=None,
            db=db,
        )
    except Exception as e:
        if request_key:
            _release_request(request_key)
        logger.error(f"Failed to save user message: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save user message: {e}"
        )

    provider = None
    if request.provider:
        try:
            from ..services.generation import get_streaming_provider

            provider = get_streaming_provider(request.provider)
        except ValueError as e:
            if request_key:
                _release_request(request_key)
            raise HTTPException(status_code=400, detail=str(e))

    async def event_generator() -> AsyncIterator[str]:
        completed = False
        disconnected = False
        errored = False
        try:
            async for event in rag_service.stream_query(
                question=request.question,
                model=request.model,
                pdf_ids=request.pdf_ids,
                db=db,
                provider=provider,
                mode=request.mode,
            ):
                if event.event == "done":
                    completed = True
                    payload = dict(event.data)
                    message = await run_in_threadpool(
                        rag_service.save_message,
                        session_id,
                        "assistant",
                        payload["answer"],
                        payload.get("sources"),
                        db,
                    )
                    payload["message_id"] = message.message_id
                    payload["session_id"] = session_id

                    timings = payload.get("timings") or {}
                    logger.info(
                        "✅ Stream complete: ttft=%ss total=%ss tokens=%s",
                        timings.get("ttft_seconds"),
                        timings.get("total_seconds"),
                        timings.get("eval_count"),
                    )
                    yield _sse("done", payload)
                    continue

                if event.event == "error":
                    # The service already reported the failure; don't add a
                    # misleading "ended without done" error after it.
                    errored = True

                if await http_request.is_disconnected():
                    logger.info(
                        "Client disconnected; cancelling upstream generation"
                    )
                    disconnected = True
                    break

                yield _sse(event.event, event.data)

            if not completed and not disconnected and not errored:
                logger.error("Stream ended without a done event")
                yield _sse(
                    "error",
                    {
                        "error": (
                            "The model stream ended before the answer was "
                            "completed."
                        )
                    },
                )
        except asyncio.CancelledError:
            logger.info("Streaming request cancelled; upstream closed")
            raise
        except Exception as e:
            logger.error(f"❌ Streaming query failed: {e}")
            yield _sse("error", {"error": f"Query failed: {e}"})
        finally:
            if request_key:
                _release_request(request_key)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Get chat history for a session."""
    messages = rag_service.get_session_messages(session_id, db)
    return [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "sources": msg.sources,
            "timestamp": msg.timestamp,
        }
        for msg in messages
    ]
