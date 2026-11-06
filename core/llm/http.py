"""Plain-HTTP LLM adapters (no vendor SDKs).

One client implements every provider documented in ``ARCHITECTURE.md`` §6.3:
Groq, OpenAI, Mistral, Gemini, and Anthropic. The differences are confined to
request shape, auth header, and response parsing; retries and fallback live in
``core.llm.base.RetryingClient``.

The API key never appears in an exception message or a response object. HTTP
errors are reduced to a status + short body snippet with the key redacted.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from config.settings import Settings
from core.llm.base import (
    LLMError,
    LLMResponse,
    Message,
    RetryingClient,
)

OPENAI_COMPATIBLE: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {*OPENAI_COMPATIBLE, "gemini", "anthropic"}
)


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _short(text: str, limit: int = 500) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit]


def _join_system(messages: list[Message]) -> str:
    return "\n\n".join(message.content for message in messages if message.role == "system")


def _non_system(messages: list[Message]) -> list[Message]:
    return [message for message in messages if message.role != "system"]


class HttpLLMClient:
    """Synchronous provider client with deterministic request/response shapes."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        provider = (provider or "").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise LLMError(
                f"unsupported LLM provider: {provider or '<empty>'}",
                provider=provider,
            )
        if not model:
            raise LLMError("LLM model id is empty", provider=provider)
        if not api_key:
            raise LLMError("LLM API key is missing", provider=provider)
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._transport = transport

    # -- public API ------------------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> LLMResponse:
        url, headers, body = self._request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        try:
            with httpx.Client(
                timeout=timeout_s,
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                response = client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{self.provider} timed out after {timeout_s:.0f}s",
                provider=self.provider,
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{self.provider} network error: {_short(str(exc))}",
                provider=self.provider,
                retryable=True,
            ) from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            retryable = response.status_code in {408, 409, 425, 429} or (
                response.status_code >= 500
            )
            raise LLMError(
                f"{self.provider} HTTP {response.status_code}: "
                f"{_redact(_short(response.text), self._api_key)}",
                provider=self.provider,
                status_code=response.status_code,
                retryable=retryable,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(
                f"{self.provider} returned invalid JSON",
                provider=self.provider,
                retryable=False,
            ) from exc

        try:
            text, usage, finish_reason = self._parse(payload)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(
                f"{self.provider} returned an unexpected response shape",
                provider=self.provider,
                retryable=False,
            ) from exc

        if not text.strip():
            raise LLMError(
                f"{self.provider} returned an empty completion",
                provider=self.provider,
                retryable=False,
            )
        return LLMResponse(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            latency_ms=elapsed_ms,
            usage=usage,
            finish_reason=finish_reason,
        )

    # -- request shapes --------------------------------------------------
    def _request(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        if self.provider in OPENAI_COMPATIBLE:
            base = OPENAI_COMPATIBLE[self.provider]
            body = {
                "model": self.model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            return (
                f"{base}/chat/completions",
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                body,
            )

        if self.provider == "gemini":
            contents = [
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.content}],
                }
                for message in _non_system(messages)
            ]
            body = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            system = _join_system(messages)
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            return (
                f"{GEMINI_BASE}/models/{self.model}:generateContent?key={self._api_key}",
                {"Content-Type": "application/json"},
                body,
            )

        # anthropic
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in _non_system(messages)
            ],
        }
        system = _join_system(messages)
        if system:
            body["system"] = system
        return (
            ANTHROPIC_URL,
            {
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            body,
        )

    # -- response shapes -------------------------------------------------
    def _parse(
        self, payload: dict[str, Any]
    ) -> tuple[str, dict[str, int], str]:
        if self.provider == "gemini":
            candidate = payload["candidates"][0]
            parts = candidate["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
            usage = {
                "prompt_tokens": int(payload.get("usageMetadata", {}).get("promptTokenCount", 0)),
                "completion_tokens": int(
                    payload.get("usageMetadata", {}).get("candidatesTokenCount", 0)
                ),
            }
            return text, usage, str(candidate.get("finishReason", ""))

        if self.provider == "anthropic":
            blocks = payload["content"]
            text = "".join(
                block.get("text", "")
                for block in blocks
                if block.get("type") == "text"
            )
            usage = {
                "prompt_tokens": int(payload.get("usage", {}).get("input_tokens", 0)),
                "completion_tokens": int(payload.get("usage", {}).get("output_tokens", 0)),
            }
            return text, usage, str(payload.get("stop_reason", ""))

        choice = payload["choices"][0]
        content = choice["message"].get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        usage = {
            "prompt_tokens": int(payload.get("usage", {}).get("prompt_tokens", 0)),
            "completion_tokens": int(payload.get("usage", {}).get("completion_tokens", 0)),
        }
        return str(content or ""), usage, str(choice.get("finish_reason", ""))


def build_client(
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> HttpLLMClient:
    """Build one adapter, raising ``LLMError`` when the key is absent."""
    active_provider = (provider or settings.llm.provider).strip().lower()
    key = settings.api_key(active_provider)
    if not key:
        raise LLMError(
            f"no API key configured for provider '{active_provider}'",
            provider=active_provider,
        )
    return HttpLLMClient(
        active_provider,
        model or settings.llm.model,
        key,
        transport=transport,
    )


def build_retrying_client(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> RetryingClient | None:
    """Return ``None`` when no provider key exists, so the UI can fall back.

    The fallback provider reuses ``settings.llm.model``. That is only useful
    when both providers expose the same model id; otherwise leave
    ``llm.fallback_provider`` unset and use one provider.
    """
    primary_key = settings.api_key(settings.llm.provider)
    if not primary_key:
        return None

    primary = build_client(settings, transport=transport)
    fallback: HttpLLMClient | None = None
    fallback_provider = settings.llm.fallback_provider
    if (
        fallback_provider
        and fallback_provider != settings.llm.provider
        and settings.api_key(fallback_provider)
    ):
        fallback = build_client(
            settings,
            provider=fallback_provider,
            transport=transport,
        )
    return RetryingClient(
        primary,
        fallback=fallback,
        max_retries=settings.llm.max_retries,
        backoff_s=0.4,
    )


__all__ = [
    "ANTHROPIC_VERSION",
    "GEMINI_BASE",
    "HttpLLMClient",
    "OPENAI_COMPATIBLE",
    "SUPPORTED_PROVIDERS",
    "build_client",
    "build_retrying_client",
]
