"""Typed configuration for the Groww MF FAQ Assistant.

One place loads ``config/config.yaml`` + ``.env`` and returns a frozen
``Settings`` object. No other module may read environment variables or open the
YAML file directly (tests excepted).

Precedence
----------
defaults  <  config/config.yaml  <  environment (.env then os.environ)  <  explicit args

The API keys are kept in ``Settings.api_keys`` (never logged, never hashed) and
are exposed only through ``Settings.api_key()``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

#: provider -> environment variable that holds its API key
PROVIDER_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

#: Stage artifacts, in pipeline order. Created on load so every stage can write.
DATA_SUBDIRS: tuple[str, ...] = (
    "01_raw",
    "02_clean",
    "03_chunks",
    "04_embeddings",
    "05_chroma",
    "06_retrieval",
    "07_answers",
)

_COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,62}$")
_ENV_PREFIX_OVERRIDES = {
    "LLM_PROVIDER": ("llm", "provider"),
    "LLM_MODEL": ("llm", "model"),
    "APP_MODE": ("app", "mode"),
    # Container hosts such as Vercel expose a read-only image filesystem.
    # Point the writable stage directories at /tmp (or another mounted volume).
    "GROWW_DATA_DIR": ("paths", "data_dir"),
    "GROWW_CHROMA_DIR": ("vector_store", "persist_dir"),
    "GROWW_TRACE_DIR": ("logging", "trace_dir"),
}


class ConfigError(RuntimeError):
    """Configuration is missing, malformed, or contradictory.

    The message always names the offending key path (e.g.
    ``retrieval.min_score``) so the fix is obvious without a traceback.
    """


class _Section(BaseModel):
    """Base for config sections: strict, immutable, typo-rejecting."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AppCfg(_Section):
    name: str = "Groww Help FAQ Assistant"
    mode: Literal["facts_only", "debug"] = "facts_only"
    max_input_chars: int = Field(default=300, ge=20, le=2000)


class PathsCfg(_Section):
    data_dir: Path = Path("data")
    sources_csv: Path = Path("sources/sources.csv")
    prompt_dir: Path = Path("prompts")


class CorpusCfg(_Section):
    allowed_domains: list[str] = Field(min_length=1)
    user_agent: str = Field(min_length=1)
    request_delay_seconds: float = Field(default=1.5, ge=0, le=60)
    timeout_seconds: float = Field(default=20, gt=0, le=120)
    max_pages: int = Field(default=5, ge=1, le=100)
    save_raw_html: bool = True
    respect_robots: bool = True
    min_clean_chars: int = Field(default=200, ge=1, le=20_000)

    @field_validator("allowed_domains")
    @classmethod
    def _clean_domains(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            domain = value.strip().lower()
            if not domain:
                raise ValueError("empty domain is not allowed")
            if "://" in domain or "/" in domain or ":" in domain:
                raise ValueError(
                    f"use a bare host such as 'groww.in', not a URL ('{value}')"
                )
            if domain not in cleaned:
                cleaned.append(domain)
        return cleaned


class ChunkingCfg(_Section):
    strategy: Literal["structure_aware_hybrid"] = "structure_aware_hybrid"
    atom_max_chars: int = Field(default=320, ge=80, le=2000)
    window_chars: int = Field(default=700, ge=200, le=4000)
    window_overlap: int = Field(default=120, ge=0, le=2000)
    min_chunk_chars: int = Field(default=60, ge=0, le=1000)

    @model_validator(mode="after")
    def _sane_sizes(self) -> ChunkingCfg:
        if self.window_overlap >= self.window_chars:
            raise ValueError("window_overlap must be smaller than window_chars")
        if self.min_chunk_chars >= self.window_chars:
            raise ValueError("min_chunk_chars must be smaller than window_chars")
        if self.atom_max_chars < self.min_chunk_chars:
            raise ValueError("atom_max_chars must be >= min_chunk_chars")
        return self


class EmbeddingCfg(_Section):
    model: str = Field(min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    batch_size: int = Field(default=64, ge=1, le=1024)
    device: Literal["cpu", "cuda"] = "cpu"
    normalize: bool = True


class VectorStoreCfg(_Section):
    provider: Literal["chroma"] = "chroma"
    persist_dir: Path = Path("data/05_chroma")
    collection: str = Field(min_length=3, max_length=63)
    distance: Literal["cosine"] = "cosine"
    reset_on_rebuild: bool = True
    batch_size: int = Field(default=256, ge=1, le=2048)

    @field_validator("collection")
    @classmethod
    def _collection_name(cls, value: str) -> str:
        if not _COLLECTION_RE.match(value):
            raise ValueError(
                "must be 3-63 chars of lowercase letters, digits, '-' or '_'"
            )
        return value


class RetrievalCfg(_Section):
    top_k: int = Field(default=6, ge=1, le=50)
    candidate_k: int = Field(default=24, ge=1, le=200)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    global_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    lexical_fallback: bool = True

    @model_validator(mode="after")
    def _candidates_cover_top_k(self) -> RetrievalCfg:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        if self.global_min_score < self.min_score:
            raise ValueError("global_min_score must be >= min_score")
        return self


class LLMCfg(_Section):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    max_tokens: int = Field(default=220, ge=8, le=4096)
    timeout_seconds: float = Field(default=30, gt=0, le=180)
    max_retries: int = Field(default=2, ge=0, le=5)
    fallback_provider: str | None = None

    @field_validator("provider")
    @classmethod
    def _provider_key(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("fallback_provider")
    @classmethod
    def _fallback_key(cls, value: str | None) -> str | None:
        return value.strip().lower() if value else None

    @field_validator("model")
    @classmethod
    def _model_id(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("model id must not be empty")
        return model


class AnswerCfg(_Section):
    max_sentences: int = Field(default=3, ge=1, le=3)
    require_citation: bool = True
    show_last_updated: bool = True


class PiiCfg(_Section):
    enabled: bool = True
    redact_in_logs: bool = True


class DebugCfg(_Section):
    save_prompts: bool = True
    inspector: bool = True
    index_stale_days: int = Field(default=30, ge=1, le=3650)


class LoggingCfg(_Section):
    trace_dir: Path = Path("data/07_answers")
    log_retrieval: bool = True


class Settings(_Section):
    app: AppCfg
    paths: PathsCfg
    corpus: CorpusCfg
    chunking: ChunkingCfg
    embedding: EmbeddingCfg
    vector_store: VectorStoreCfg
    retrieval: RetrievalCfg
    llm: LLMCfg
    answer: AnswerCfg
    pii: PiiCfg
    debug: DebugCfg
    logging: LoggingCfg
    api_keys: dict[str, str] = Field(default_factory=dict, repr=False)
    config_hash: str = ""

    # -- secrets ---------------------------------------------------------
    def api_key(self, provider: str | None = None) -> str | None:
        """Return the key for ``provider`` (default: configured provider)."""
        return self.api_keys.get(provider or self.llm.provider)

    def masked(self) -> dict[str, Any]:
        """Config dump with secrets and hashes removed; safe to log."""
        data = self.model_dump(mode="json", exclude={"api_keys", "config_hash"})
        data["api_keys_present"] = sorted(self.api_keys)
        return data

    # -- identity --------------------------------------------------------
    def index_fingerprint(self) -> str:
        """Hash of everything that changes the vector index.

        Retrieval/answer/LLM changes do NOT change this, so tuning them does
        not force a re-index. Chunking, embedding, or collection changes do.
        """
        payload = {
            "chunking": self.chunking.model_dump(mode="json"),
            "embedding": self.embedding.model_dump(mode="json"),
            "vector_store": {
                "provider": self.vector_store.provider,
                "collection": self.vector_store.collection,
                "distance": self.vector_store.distance,
            },
        }
        return _sha256(payload)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_settings(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    ensure_dirs: bool = True,
    require_provider_key: bool = True,
) -> Settings:
    """Load and validate configuration.

    Parameters
    ----------
    config_path:
        Defaults to ``config/config.yaml`` next to this file.
    env:
        Explicit environment mapping (tests use ``{}``). When ``None``, reads
        ``.env`` and then ``os.environ`` (os.environ wins).
    ensure_dirs:
        Create ``data/01_raw`` ... ``data/07_answers``.
    require_provider_key:
        Fail if the selected provider has no API key. Tests that never call an
        LLM pass ``False``.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _read_yaml(path)
    env_map = dict(env) if env is not None else _read_env()
    raw = _apply_env_overrides(raw, env_map)

    try:
        settings = Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation(exc)) from None

    keys = {
        provider: env_map[var]
        for provider, var in PROVIDER_ENV.items()
        if env_map.get(var)
    }
    settings = settings.model_copy(update={"api_keys": keys})
    settings = _resolve_paths(settings)

    _check_semantics(settings, require_provider_key=require_provider_key)

    if ensure_dirs:
        ensure_data_dirs(settings)

    settings = settings.model_copy(update={"config_hash": _config_hash(settings)})
    return settings


def ensure_data_dirs(settings: Settings) -> None:
    """Create the stage folders and prove they are writable."""
    root = settings.paths.data_dir
    try:
        root.mkdir(parents=True, exist_ok=True)
        for name in DATA_SUBDIRS:
            (root / name).mkdir(parents=True, exist_ok=True)
        probe = root / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise ConfigError(f"paths.data_dir is not writable ({root}): {exc}") from None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _read_env() -> dict[str, str]:
    merged: dict[str, str] = {
        key: value
        for key, value in dotenv_values(PROJECT_ROOT / ".env").items()
        if value is not None
    }
    merged.update({k: v for k, v in os.environ.items() if v is not None})
    return merged


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    data = copy.deepcopy(raw)
    for var, (section, key) in _ENV_PREFIX_OVERRIDES.items():
        value = env.get(var)
        if value:
            data.setdefault(section, {})[key] = value
    return data


def _resolve_paths(settings: Settings) -> Settings:
    def absolute(value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    return settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={
                    "data_dir": absolute(settings.paths.data_dir),
                    "sources_csv": absolute(settings.paths.sources_csv),
                    "prompt_dir": absolute(settings.paths.prompt_dir),
                }
            ),
            "vector_store": settings.vector_store.model_copy(
                update={"persist_dir": absolute(settings.vector_store.persist_dir)}
            ),
            "logging": settings.logging.model_copy(
                update={"trace_dir": absolute(settings.logging.trace_dir)}
            ),
        }
    )


def _check_semantics(settings: Settings, *, require_provider_key: bool) -> None:
    provider = settings.llm.provider
    if provider not in PROVIDER_ENV:
        known = ", ".join(sorted(PROVIDER_ENV))
        raise ConfigError(f"llm.provider '{provider}' is unknown; use one of: {known}")

    fallback = settings.llm.fallback_provider
    if fallback and fallback not in PROVIDER_ENV:
        known = ", ".join(sorted(PROVIDER_ENV))
        raise ConfigError(
            f"llm.fallback_provider '{fallback}' is unknown; use one of: {known}"
        )

    if require_provider_key:
        if not settings.api_key(provider):
            raise ConfigError(
                f"missing API key: set {PROVIDER_ENV[provider]} in .env "
                f"for llm.provider '{provider}'"
            )
        if fallback and not settings.api_key(fallback):
            raise ConfigError(
                f"llm.fallback_provider '{fallback}' needs {PROVIDER_ENV[fallback]} in .env"
            )

    if not _COLLECTION_RE.match(settings.vector_store.collection):
        raise ConfigError(
            "vector_store.collection must be 3-63 chars of lowercase letters, "
            "digits, '-' or '_'"
        )


def _config_hash(settings: Settings) -> str:
    """Stable identity of the non-secret, non-machine-specific config.

    Paths and debug flags are excluded so two machines with different checkout
    directories agree; the data-dir path must not invalidate an index.
    """
    payload = settings.model_dump(mode="json", exclude={"api_keys", "config_hash", "paths", "debug"})
    payload.get("vector_store", {}).pop("persist_dir", None)
    return _sha256(payload)


def _sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _format_validation(exc: ValidationError) -> str:
    lines = ["invalid configuration:"]
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {where}: {error['msg']}")
    return "\n".join(lines)


__all__ = [
    "AppCfg",
    "AnswerCfg",
    "ChunkingCfg",
    "ConfigError",
    "CorpusCfg",
    "DATA_SUBDIRS",
    "DebugCfg",
    "EmbeddingCfg",
    "LLMCfg",
    "LoggingCfg",
    "PathsCfg",
    "PiiCfg",
    "PROJECT_ROOT",
    "PROVIDER_ENV",
    "RetrievalCfg",
    "Settings",
    "VectorStoreCfg",
    "ensure_data_dirs",
    "load_settings",
]
