"""Inspect Step-7 retrieval without generating an answer.

Usage:
    python scripts/retrieve.py "when will my stocks reach my demat account"
    python scripts/retrieve.py --json "how do i change my groww password"
    python scripts/retrieve.py --no-trace "account opening charges"

The command loads the verified Chroma index, prints the resolved plan, each
filter level's score, the MMR picks, and the gate result. It never calls an LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import ConfigError, Settings, load_settings  # noqa: E402
from core.retriever import Retriever  # noqa: E402
from core.store import IndexStale, StoreError  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve and retrieve facts from the five-page corpus."
    )
    parser.add_argument("question", nargs="*", help="question text")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--json", action="store_true", help="print the retrieval result as JSON"
    )
    parser.add_argument(
        "--no-trace", action="store_true", help="do not write data/06_retrieval"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="override retrieval.top_k for this run",
    )
    args = parser.parse_args(argv)
    if not args.question:
        parser.error("provide a question, e.g. \"account opening charges\"")
    return args


def _with_top_k(settings: Settings, top_k: int | None) -> Settings:
    if top_k is None:
        return settings
    if top_k < 1:
        raise ConfigError("--top-k must be >= 1")
    if top_k > settings.retrieval.candidate_k:
        raise ConfigError(
            f"--top-k ({top_k}) must be <= retrieval.candidate_k "
            f"({settings.retrieval.candidate_k})"
        )
    retrieval = settings.retrieval.model_copy(update={"top_k": top_k})
    return settings.model_copy(update={"retrieval": retrieval})


def _trace_chunk(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "score": round(chunk.score, 6),
        "source": chunk.source,
        "metadata": dict(chunk.metadata),
    }


def _payload(result) -> dict:
    plan = result.plan
    return {
        "qid": result.qid,
        "question": plan.question,
        "intent": plan.intent,
        "scheme": plan.scheme,
        "scheme_slug": plan.scheme_slug,
        "field": plan.field,
        "filters": plan.filters,
        "ambiguous_scheme": plan.ambiguous_scheme,
        "ambiguous_field": plan.ambiguous_field,
        "notes": plan.notes,
        "filter_level": result.filter_level,
        "best_score": result.best_score,
        "gate_passed": result.gate_passed,
        "selected": [_trace_chunk(chunk) for chunk in result.selected],
        "trace_path": result.trace_path,
    }


def _print_human(result) -> None:
    plan = result.plan
    print(f"qid          : {result.qid}")
    print(f"intent       : {plan.intent}")
    print(
        "scheme       : "
        f"{plan.scheme or '-'}"
        + (f" ({plan.scheme_slug})" if plan.scheme_slug else "")
    )
    print(f"field        : {plan.field or '-'}")
    print(f"filters      : {plan.filters or {}}")
    print(f"filter level : {result.filter_level}")
    print(
        f"gate         : {'PASS' if result.gate_passed else 'FAIL'}"
        f" (best {result.best_score:.3f})"
    )
    if plan.notes:
        print(f"notes        : {', '.join(plan.notes)}")
    print()

    if not result.selected:
        print("No chunks selected.")
    else:
        for rank, chunk in enumerate(result.selected, 1):
            metadata = chunk.metadata
            print(
                f"{rank}. [{chunk.source}] {chunk.score:.3f} "
                f"{metadata.get('scheme_slug', '?')} / "
                f"{metadata.get('field') or metadata.get('section_title') or '?'}"
            )
            print(f"   {chunk.text}")

    if result.trace_path:
        print(f"\ntrace        : {result.trace_path}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)
    question = " ".join(args.question)

    try:
        settings = load_settings(
            args.config, ensure_dirs=True, require_provider_key=False
        )
        settings = _with_top_k(settings, args.top_k)
        retriever = Retriever.open(settings, write_trace=not args.no_trace)
        result = retriever.retrieve(question)
    except (ConfigError, IndexStale, StoreError) as exc:
        print(f"RETRIEVAL ERROR\n{exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_payload(result), indent=2, ensure_ascii=False))
    else:
        _print_human(result)
    return 0 if result.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
