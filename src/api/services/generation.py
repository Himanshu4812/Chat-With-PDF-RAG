"""Streaming chat provider abstraction for Ollama and Groq Cloud.

Provides genuine token streaming, safe connection lifecycles, and typed
exceptions for rate-limits, auth, timeouts, and network failures.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

import httpx

try:
    from ..config import settings
except (ImportError, ValueError):
    from src.api.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Provider Exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class ProviderAuthError(ProviderError):
    """Authentication or authorization failure (e.g. invalid API key, HTTP 401)."""
    pass


class ProviderRateLimitError(ProviderError):
    """Rate or token limit exceeded (e.g. HTTP 429)."""
    pass


class ProviderTimeoutError(ProviderError):
    """Provider request or stream timed out."""
    pass


class ProviderNetworkError(ProviderError):
    """Network connection failure, disconnection, or unreachable host."""
    pass


class ProviderServerError(ProviderError):
    """Provider internal server error (e.g. HTTP 500, 503)."""
    pass


class ProviderNotFoundError(ProviderError):
    """Model or resource not found on provider (e.g. HTTP 404)."""
    pass


# ---------------------------------------------------------------------------
# Chunk and Protocol
# ---------------------------------------------------------------------------


@dataclass
class ChatChunk:
    """A single chunk from a streaming chat provider."""

    content: str = ""
    thinking: str = ""
    done: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


class StreamingChatProvider(Protocol):
    """Provider interface shared by Ollama and Groq streaming."""

    name: str

    def stream_chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        think: bool = False,
    ) -> AsyncIterator[ChatChunk]:
        """Yield ChatChunk objects until the provider signals done."""


# ---------------------------------------------------------------------------
# Ollama Streaming Provider
# ---------------------------------------------------------------------------


class OllamaStreamingProvider:
    """Genuine token streaming from Ollama's /api/chat endpoint."""

    name = "ollama"

    def __init__(
        self,
        host: Optional[str] = None,
        timeout_seconds: float = 300.0,
    ):
        self.host = (host or settings.OLLAMA_HOST).rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def stream_chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        think: bool = False,
    ) -> AsyncIterator[ChatChunk]:
        # If a Groq model name was passed, use default local chat model instead
        local_model = model
        if not local_model or "llama-3.3" in local_model or "llama-3.1" in local_model:
            local_model = getattr(settings, "DEFAULT_CHAT_MODEL", "phi4-mini:3.8b")

        payload: Dict[str, Any] = {
            "model": local_model,
            "messages": messages,
            "stream": True,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": settings.NUM_CTX,
                "num_predict": settings.NUM_PREDICT,
                "temperature": settings.LLM_TEMPERATURE,
            },
        }
        if think:
            payload["think"] = True

        timeout = httpx.Timeout(self.timeout_seconds, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{self.host}/api/chat", json=payload
                ) as response:
                    if response.status_code == 404:
                        body = await response.aread()
                        detail = body.decode("utf-8", errors="replace")[:300]
                        raise ProviderNotFoundError(
                            f"Ollama model '{local_model}' not found: {detail}"
                        )
                    elif response.status_code >= 500:
                        body = await response.aread()
                        detail = body.decode("utf-8", errors="replace")[:300]
                        raise ProviderServerError(
                            f"Ollama server error {response.status_code}: {detail}"
                        )
                    elif response.status_code >= 400:
                        body = await response.aread()
                        detail = body.decode("utf-8", errors="replace")[:300]
                        raise ProviderError(
                            f"Ollama error {response.status_code}: {detail}"
                        )

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping malformed Ollama stream line: %s",
                                line[:200],
                            )
                            continue

                        if data.get("error"):
                            err_msg = str(data["error"])
                            if "not found" in err_msg.lower():
                                raise ProviderNotFoundError(f"Ollama error: {err_msg}")
                            raise ProviderError(f"Ollama error: {err_msg}")

                        message = data.get("message") or {}
                        yield ChatChunk(
                            content=message.get("content") or "",
                            thinking=message.get("thinking") or "",
                            done=bool(data.get("done")),
                            metrics=data,
                        )
        except httpx.ConnectError as e:
            raise ProviderNetworkError(
                f"Failed to connect to Ollama at {self.host}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self.timeout_seconds}s"
            ) from e
        finally:
            logger.info("Ollama stream closed (upstream connection released)")


# ---------------------------------------------------------------------------
# Groq Streaming Provider
# ---------------------------------------------------------------------------


class GroqStreamingProvider:
    """Genuine token streaming from Groq Cloud API."""

    name = "groq"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.api_key = api_key or getattr(settings, "GROQ_API_KEY", None)
        self.default_model = (
            model or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-20b")
        )
        self.base_url = (
            base_url
            or getattr(settings, "GROQ_API_BASE", "https://api.groq.com/openai/v1")
        ).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "GROQ_TIMEOUT_SECONDS", 30.0)
        )

    def _resolve_model(self, model: str) -> str:
        """Map model name to a Groq-compatible model.

        If model passed is a local Ollama model (e.g. 'phi4-mini:3.8b', 'mistral:latest')
        or empty, use the configured Groq model.
        """
        if (
            not model
            or ":" in model
            or "phi" in model.lower()
            or "mistral" in model.lower()
            or "nomic" in model.lower()
        ):
            return self.default_model
        return model

    async def stream_chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        think: bool = False,
    ) -> AsyncIterator[ChatChunk]:
        if not self.api_key:
            raise ProviderAuthError(
                "Groq API key not configured. Set GROQ_API_KEY in backend environment."
            )

        groq_model = self._resolve_model(model)
        payload: Dict[str, Any] = {
            "model": groq_model,
            "messages": messages,
            "stream": True,
            "temperature": getattr(settings, "GROQ_TEMPERATURE", 0.1),
            "max_tokens": getattr(settings, "GROQ_MAX_TOKENS", 512),
            "stream_options": {"include_usage": True},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        timeout = httpx.Timeout(self.timeout_seconds, connect=10.0)
        usage_metrics: Dict[str, Any] = {}
        yielded_done = False

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code == 401:
                        raise ProviderAuthError(
                            "Invalid or unauthorized Groq API key (HTTP 401)"
                        )
                    elif response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        msg = (
                            f"Groq rate limit exceeded (HTTP 429). Retry after: {retry_after}s"
                            if retry_after
                            else "Groq rate limit exceeded (HTTP 429). Free-tier limit reached."
                        )
                        raise ProviderRateLimitError(msg)
                    elif response.status_code >= 500:
                        body = await response.aread()
                        detail = body.decode("utf-8", errors="replace")[:300]
                        raise ProviderServerError(
                            f"Groq server error (HTTP {response.status_code}): {detail}"
                        )
                    elif response.status_code >= 400:
                        body = await response.aread()
                        detail = body.decode("utf-8", errors="replace")[:300]
                        raise ProviderError(
                            f"Groq API error {response.status_code}: {detail}"
                        )

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                        elif line.startswith("data:"):
                            data_str = line[5:].strip()
                        else:
                            continue

                        if data_str == "[DONE]":
                            yielded_done = True
                            yield ChatChunk(
                                content="",
                                done=True,
                                metrics={
                                    "usage": usage_metrics,
                                    "model": groq_model,
                                    "provider": "groq",
                                },
                            )
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping malformed Groq stream line: %s",
                                line[:200],
                            )
                            continue

                        if data.get("error"):
                            err = data["error"]
                            msg = (
                                err.get("message", str(err))
                                if isinstance(err, dict)
                                else str(err)
                            )
                            raise ProviderError(f"Groq streaming error: {msg}")

                        choices = data.get("choices") or []
                        content = ""
                        thinking = ""
                        finish_reason = None
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta") or {}
                            content = delta.get("content") or ""
                            thinking = (
                                delta.get("reasoning")
                                or delta.get("thinking")
                                or ""
                            )
                            finish_reason = choice.get("finish_reason")

                        usage = data.get("usage")
                        if usage:
                            usage_metrics.update(usage)

                        is_done = bool(finish_reason in ("stop", "length"))
                        if is_done:
                            yielded_done = True

                        if content or thinking or is_done:
                            yield ChatChunk(
                                content=content,
                                thinking=thinking,
                                done=is_done,
                                metrics={
                                    "usage": usage_metrics,
                                    "model": groq_model,
                                    "provider": "groq",
                                },
                            )

                    if not yielded_done:
                        yield ChatChunk(
                            content="",
                            done=True,
                            metrics={
                                "usage": usage_metrics,
                                "model": groq_model,
                                "provider": "groq",
                            },
                        )
        except httpx.ConnectError as e:
            raise ProviderNetworkError(
                f"Failed to connect to Groq API: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                f"Groq API timed out after {self.timeout_seconds}s"
            ) from e
        finally:
            logger.info("Groq stream closed (upstream connection released)")


def get_streaming_provider(
    provider: Optional[str] = None,
) -> StreamingChatProvider:
    """Resolve the configured chat provider."""
    name = (provider or getattr(settings, "CHAT_PROVIDER", "ollama")).lower()
    if name == "ollama":
        return OllamaStreamingProvider()
    elif name == "groq":
        return GroqStreamingProvider()
    raise ValueError(f"Unsupported streaming provider: {name}")
