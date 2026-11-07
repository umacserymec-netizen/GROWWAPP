"""Optional local FastAPI surface (see ``ARCHITECTURE.md`` §10.1).

The Streamlit UI calls ``core.answer`` directly. This API exists for curl/demo
use and returns only public fields on ``/ask``; prompts and raw model text stay
in ``data/07_answers`` and are available from ``/trace/{qid}`` for the local
inspector.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import Settings, load_settings
from core.answer import AnswerEngine
from core.store import (
    INDEX_META_NAME,
    IndexStale,
    load_index_meta,
    validate_index_meta,
)

app = FastAPI(title="Groww Help FAQ Assistant", version="0.1.0")

_PUBLIC_FIELDS = (
    "qid",
    "status",
    "answer_text",
    "citation_url",
    "citation_title",
    "last_updated",
    "intent",
    "scheme",
    "field",
    "chunk_ids",
    "refused",
    "refusal_reason",
    "latency_ms",
    "trace_id",
    "fallback_used",
    "fallback_reason",
    "error_code",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    include_trace: bool = False


@lru_cache(maxsize=1)
def get_engine() -> AnswerEngine:
    settings = load_settings(require_provider_key=False)
    return AnswerEngine.open(settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings(require_provider_key=False)


def _trace(trace_dir: Path, qid: str) -> dict[str, Any] | None:
    matches = sorted(Path(trace_dir).glob(f"*_{qid}.json"))
    if not matches:
        return None
    try:
        payload = json.loads(matches[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


@app.exception_handler(IndexStale)
async def _index_stale_handler(request: Request, exc: IndexStale) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "code": "E_INDEX_STALE",
            "message": str(exc),
        },
    )


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    engine = get_engine()
    record = engine.ask(request.question)
    payload = asdict(record)
    response = {key: payload.get(key) for key in _PUBLIC_FIELDS}
    if request.include_trace:
        response["trace"] = _trace(
            Path(engine.settings.paths.data_dir) / "06_retrieval",
            record.qid,
        )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    meta_ok = True
    index_age_days: int | None = None
    try:
        meta = load_index_meta(
            Path(settings.vector_store.persist_dir) / INDEX_META_NAME
        )
        validate_index_meta(settings, meta)
        try:
            fetched = datetime.fromisoformat(
                meta.newest_fetched_at.replace("Z", "+00:00")
            ).date()
            index_age_days = (date.today() - fetched).days
        except ValueError:
            index_age_days = None
    except IndexStale:
        meta_ok = False
    return {
        "status": "ok",
        "index_meta_ok": meta_ok,
        "index_age_days": index_age_days,
        # A provider is "reachable" only in the sense that a key is configured;
        # health must not spend a model call or leak the key.
        "provider_reachable": bool(settings.api_key()),
    }


@app.get("/trace/{qid}")
def trace(qid: str) -> dict[str, Any]:
    settings = get_settings()
    answer_path = Path(settings.logging.trace_dir) / f"{qid}.json"
    try:
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        answer = None
    return {
        "qid": qid,
        "answer": answer,
        "retrieval": _trace(
            Path(settings.paths.data_dir) / "06_retrieval",
            qid,
        ),
    }


if __name__ == "__main__":  # ``python -m scripts.api`` → :8000 (ARCHITECTURE §10.2)
    import uvicorn

    uvicorn.run("scripts.api:app", host="127.0.0.1", port=8000, reload=False)
