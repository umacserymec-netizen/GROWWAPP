"""Provider-neutral LLM types and the retry/fallback wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)


class LLMError(RuntimeError):
    """A provider refused, timed out, or returned an unusable payload."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        retryable: bool = False,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts or []


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> LLMResponse: ...


class RetryingClient:
    """One retry budget across a primary provider and an optional fallback."""

    def __init__(
        self,
        primary: LLMClient,
        *,
        fallback: LLMClient | None = None,
        max_retries: int = 1,
        backoff_s: float = 0.4,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max(0, max_retries)
        self.backoff_s = max(0.0, backoff_s)

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> LLMResponse:
        attempts: list[dict[str, Any]] = []
        clients = [self.primary]
        if self.fallback is not None:
            clients.append(self.fallback)

        last_error: LLMError | None = None
        for client in clients:
            for attempt_no in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    response = client.complete(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                    )
                except LLMError as exc:
                    elapsed = int((time.perf_counter() - started) * 1000)
                    attempts.append(
                        {
                            "provider": exc.provider or _client_provider(client),
                            "attempt": attempt_no + 1,
                            "latency_ms": elapsed,
                            "status_code": exc.status_code,
                            "error": str(exc)[:240],
                        }
                    )
                    last_error = exc
                    if not exc.retryable:
                        break
                    if attempt_no < self.max_retries:
                        time.sleep(self.backoff_s)
                    continue

                elapsed = int((time.perf_counter() - started) * 1000)
                attempts.append(
                    {
                        "provider": response.provider,
                        "attempt": attempt_no + 1,
                        "latency_ms": elapsed,
                        "status_code": 200,
                        "error": None,
                    }
                )
                response.attempts = attempts
                return response

        raise LLMError(
            "all LLM providers failed",
            provider=last_error.provider if last_error else "",
            status_code=last_error.status_code if last_error else None,
            retryable=False,
            attempts=attempts,
        ) from last_error


def _client_provider(client: LLMClient) -> str:
    return str(getattr(client, "provider", ""))


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "RetryingClient",
]
