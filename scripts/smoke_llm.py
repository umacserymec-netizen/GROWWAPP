"""Stage-1 gate: prove the configured LLM provider + model actually answer.

This is intentionally a standalone script (plain HTTP, no SDK, no project
dependencies beyond httpx). The production adapters in ``core/llm/`` are built
in Stage 6 and reuse the same request shapes.

Usage:
    python scripts/smoke_llm.py
    python scripts/smoke_llm.py --models      # list model ids for the provider
    python scripts/smoke_llm.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROVIDER_ENV, ConfigError, load_settings  # noqa: E402

PROMPT = "Reply with the single word: ok"
OPENAI_COMPATIBLE = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
}
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def _chat_request(provider: str, model: str, key: str) -> tuple[str, dict, dict]:
    if provider in OPENAI_COMPATIBLE:
        url = f"{OPENAI_COMPATIBLE[provider]}/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": 8,
        }
        return url, headers, body

    if provider == "gemini":
        url = f"{GEMINI_BASE}/models/{model}:generateContent?key={key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 8},
        }
        return url, {}, body

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        body = {
            "model": model,
            "max_tokens": 8,
            "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT}],
        }
        return url, headers, body

    raise ValueError(f"unsupported provider: {provider}")


def _models_request(provider: str, key: str) -> tuple[str, dict]:
    if provider in OPENAI_COMPATIBLE:
        return f"{OPENAI_COMPATIBLE[provider]}/models", {"Authorization": f"Bearer {key}"}
    if provider == "gemini":
        return f"{GEMINI_BASE}/models?key={key}", {}
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/models", {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
    raise ValueError(f"unsupported provider: {provider}")


def _extract_text(provider: str, data: dict) -> str:
    try:
        if provider == "gemini":
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if provider == "anthropic":
            return data["content"][0]["text"].strip()
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected response shape: {exc}") from None


def _extract_model_ids(provider: str, data: dict) -> list[str]:
    if provider == "gemini":
        return [item.get("name", "").removeprefix("models/") for item in data.get("models", [])]
    return [item.get("id", "") for item in data.get("data", []) if item.get("id")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ping the configured LLM provider.")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--models", action="store_true", help="list available model ids")
    args = parser.parse_args()

    try:
        settings = load_settings(args.config, require_provider_key=False)
    except ConfigError as exc:
        print(f"CONFIG ERROR\n{exc}", file=sys.stderr)
        return 2

    provider = settings.llm.provider
    model = settings.llm.model
    key = settings.api_key(provider)

    if not key:
        env_var = PROVIDER_ENV.get(provider, "<unknown>")
        print(
            f"No API key for provider '{provider}'.\n"
            f"Set {env_var} in .env (see .env.example), then re-run.",
            file=sys.stderr,
        )
        return 1

    timeout = settings.llm.timeout_seconds

    try:
        with httpx.Client(timeout=timeout) as client:
            if args.models:
                url, headers = _models_request(provider, key)
                response = client.get(url, headers=headers)
                if response.status_code >= 400:
                    print(
                        f"HTTP {response.status_code} listing models:\n"
                        f"{_redact(response.text[:600], key)}",
                        file=sys.stderr,
                    )
                    return 1
                ids = _extract_model_ids(provider, response.json())
                print(f"{provider}: {len(ids)} models")
                for model_id in sorted(ids):
                    print(f"  {model_id}")
                if model not in ids:
                    print(
                        f"\nWARN: configured model '{model}' is not in this list. "
                        f"Update LLM_MODEL in .env."
                    )
                return 0

            url, headers, body = _chat_request(provider, model, key)
            started = time.perf_counter()
            response = client.post(url, headers=headers, json=body)
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            if response.status_code >= 400:
                print(
                    f"FAILED: HTTP {response.status_code} from {provider}\n"
                    f"{_redact(response.text[:800], key)}\n"
                    f"Hint: python scripts/smoke_llm.py --models",
                    file=sys.stderr,
                )
                return 1

            try:
                text = _extract_text(provider, response.json())
            except RuntimeError as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
                return 1

    except httpx.HTTPError as exc:
        print(f"FAILED: network error talking to {provider}: {exc}", file=sys.stderr)
        return 1

    print(f"provider : {provider}")
    print(f"model    : {model}")
    print(f"latency  : {elapsed_ms} ms")
    print(f"reply    : {text!r}")
    print("OK" if "ok" in text.lower() else "OK (unexpected reply, connectivity works)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
