"""Stage 2b: turn the saved HTML into clean, sectioned text.

Groww help articles are server-rendered prose inside ``div.layout-main``. The
article title is the first ``<h1>`` that is not the site's "Customer Support"
banner; the body is a nested tree of paragraphs, ordered lists, and occasional
tables. Generic article extractors keep the nav/footer and button chrome, so
this module walks the known article container directly and keeps only the
answer text.

The scheme/AMC extractors are retained for compatibility with the original
pipeline, but this project's sources are all ``page_type=help``.

Design rules
------------
- Only text from the saved snapshot under ``data/01_raw`` is used. No network.
- Digits, ``%``, ``₹`` and dates are preserved verbatim.
- Noise sections (holdings, returns, compare, calculator, fund lists) are never
  emitted, so a later stage cannot retrieve a performance claim.
- A page with missing required facts is written for inspection but flagged
  ``ok=false`` in ``parse_report.json`` (quarantined); it must not be chunked.

Artifacts
---------
``data/02_clean/<slug>.md``    human-readable, used by the demo inspector
``data/02_clean/<slug>.json``  machine-readable ``CleanDoc``
``data/02_clean/parse_report.json``  fact probes, gaps, quarantine reasons
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from config.settings import ConfigError, Settings
from core.types import CleanDoc, Section, SourceRow
from ingest.fetch import load_sources, slug_for

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "canvas",
    "form",
    "button",
    "nav",
    "footer",
    "aside",
    "img",
)
NOISE_CLASS_RE = re.compile(
    r"(rodal|cookie|consent|watchlist|infobutton|iconbutton|advert|sponsor"
    r"|social|share|qrcode|tooltip|modal|popup|signupcta|sticky)",
    re.I,
)
BLOCK_TAGS = {
    "div",
    "section",
    "article",
    "main",
    "header",
    "footer",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "ul",
    "ol",
    "li",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
UI_NOISE_LINES = {
    "view details",
    "read more",
    "read less",
    "know more",
    "invest now",
    "add to watchlist",
    "view all",
    "show more",
    ";",
    "•",
    "-",
    "|",
}
#: Chrome that leaks into the help article container after tag stripping.
HELP_NOISE_LINES = {
    "loading...",
    "loading…",
    "was the answer helpful?",
    "was this answer helpful?",
    "check pricing",
    "check price",
    "change password",
    "customer support",
    "help and support",
    "feedback",
    "what's new",
    "en",
    "हि",
}
HELP_NOISE_PREFIXES = (
    "check pricing",
    "check pass",
    "© ",
    "version:",
    "about us pricing blog",
    "download the app",
    "contact us",
    "vaishnavi tech park",
    "sarjapur main road",
    "bengaluru",
    "karnataka",
)
HELP_DROP_FRAGMENTS = (
    "button below",
    "click the button",
    "go directly to the page",
    "click the 'check pricing' button",
    "click the 'pricing' button",
)
RISK_RE = re.compile(r"\b(low|moderate|high|very high)\s+risk\b", re.I)
LOCKIN_RE = re.compile(r"\block[\s-]?in\b", re.I)
NOISE_LINE_RE = re.compile(r"^(?:scheme information document|sid)\b", re.I)

#: label -> regex probe. ``None`` means optional for that page type.
FACT_PROBES: dict[str, dict[str, str]] = {
    "scheme": {
        "expense_ratio": r"expense ratio\s*:\s*\d+(?:\.\d+)?\s*%",
        "exit_load": r"exit load(?:\s*:\s*|\s+of\s+|\s*\n\s*)(?:nil|[^.\n]{1,120})",
        "min_sip": r"min\.?\s*for\s+sip\s*:\s*₹\s?[\d,]+",
        "min_investment": r"min\.?\s*for\s+1st\s+investment\s*:\s*₹\s?[\d,]+",
        "riskometer": r"riskometer\s*:\s*[A-Za-z ]{3,40}",
        "benchmark": r"(?:fund )?benchmark\s*:\s*[A-Za-z0-9][^\n]{2,90}",
        "lock_in": r"lock[\s-]?in\s*:\s*[^\n.]{1,60}",
        "nav": r"nav:\s*[^\n]{1,60}₹[\d,.]+",
        "aum": r"(?:aum|fund size)[^:\n]{0,30}:\s*₹[\d,.]+",
        "fund_manager": r"[A-Z][A-Za-z. ]{2,40} is the Current Fund Manager",
        "launch_date": r"(?:made available to investors on|launched by)[^\n.]{1,90}",
    },
    "amc": {
        "aum": r"(?:aum|total aum)[^:\n]{0,40}:\s*₹[\d,.]+",
        "scheme_count": r"no of schemes\s*:\s*\d+",
        "amc_age": r"amc age\s*:\s*[^\n]{1,30}",
        "incorporation_date": r"(?:amc )?incorporation date\s*:\s*[^\n]{1,40}",
        "trustee": r"trustee organisation\s*:\s*[^\n]{1,80}",
    },
}

REQUIRED_FACTS: dict[str, tuple[str, ...]] = {
    "scheme": ("expense_ratio", "exit_load", "min_sip", "riskometer", "benchmark"),
    "amc": ("aum", "incorporation_date"),
}

#: scheme slugs that must also expose the statutory lock-in
LOCKIN_SCHEMES = ("elss",)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PageReport:
    id: str
    slug: str
    url: str
    scheme: str
    page_type: str
    ok: bool
    char_len: int
    section_count: int
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    reason: str | None = None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def parse_all(settings: Settings, *, today: str | None = None) -> dict[str, Any]:
    """Parse every successfully fetched page and write ``02_clean`` artifacts."""
    raw_dir = settings.paths.data_dir / "01_raw"
    out_dir = settings.paths.data_dir / "02_clean"
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(
            f"{manifest_path} not found; run 'python -m ingest.run fetch' first"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {row.id: row for row in load_sources(settings.paths.sources_csv)}
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[PageReport] = []
    for entry in manifest:
        source = sources.get(entry["id"])
        if source is None:
            continue
        report = _parse_entry(settings, entry, source, raw_dir, out_dir)
        reports.append(report)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetched_on": today or date.today().isoformat(),
        "config_hash": settings.config_hash,
        "min_clean_chars": settings.corpus.min_clean_chars,
        "pages": [asdict(report) for report in reports],
        "summary": _summary(reports),
        "gaps": [
            {"id": report.id, "scheme": report.scheme, "missing": report.missing}
            for report in reports
            if report.missing
        ],
        "quarantined": [
            {"id": report.id, "slug": report.slug, "reason": report.reason}
            for report in reports
            if not report.ok
        ],
    }
    (out_dir / "parse_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def parse_html(
    html: str,
    *,
    scheme: str,
    source_url: str,
    source_title: str,
    page_type: str,
    fetched_at: str,
    raw_sha256: str = "",
) -> CleanDoc:
    """Pure function: HTML string -> :class:`CleanDoc`. No filesystem access."""
    soup = BeautifulSoup(html, "lxml")
    _strip_noise(soup)

    if page_type == "help":
        sections = _help_sections(soup, source_title)
    elif page_type == "scheme":
        sections = _scheme_sections(soup, scheme)
    elif page_type == "amc":
        sections = _amc_sections(soup)
    else:
        sections = _generic_sections(soup)

    sections = [section for section in sections if section.body.strip()]
    char_len = sum(len(section.title) + len(section.body) for section in sections)

    return CleanDoc(
        scheme=scheme,
        scheme_slug=slug_for(source_url),
        source_title=source_title,
        source_url=source_url,
        page_type=page_type,  # type: ignore[arg-type]
        fetched_at=fetched_at,
        sections=sections,
        raw_sha256=raw_sha256,
        char_len=char_len,
    )


def doc_to_markdown(doc: CleanDoc) -> str:
    lines = [
        f"# {doc.scheme}",
        "",
        f"- Source: {doc.source_url}",
        f"- Page type: {doc.page_type}",
        f"- Fetched: {doc.fetched_at}",
    ]
    if doc.raw_sha256:
        lines.append(f"- Raw sha256: {doc.raw_sha256}")
    for section in doc.sections:
        lines.extend(["", f"## {section.title}", "", section.body.strip()])
    return "\n".join(lines).rstrip() + "\n"


def fact_hits(doc: CleanDoc) -> dict[str, dict[str, Any]]:
    """Run the fact probes over the rendered document (not the raw HTML)."""
    probes = FACT_PROBES.get(doc.page_type, {})
    # Include section titles: the exit-load value is often just "Nil" in the body.
    corpus = "\n".join(f"{section.title}\n{section.body}" for section in doc.sections)
    hits: dict[str, dict[str, Any]] = {}
    for name, pattern in probes.items():
        match = re.search(pattern, corpus, re.IGNORECASE)
        hits[name] = {
            "found": bool(match),
            "snippet": _clean(match.group(0))[:200] if match else "",
        }
    return hits


# ---------------------------------------------------------------------------
# per-page extraction
# ---------------------------------------------------------------------------


def _help_sections(soup: BeautifulSoup, fallback_title: str) -> list[Section]:
    """Extract one help article as a single titled prose section.

    Groww renders the whole answer inside ``div.layout-main``. The body is a
    deep tree of wrapper ``div``s around ``font``/``span`` text, with occasional
    ``ol`` step lists. Walking only leaf blocks keeps paragraph boundaries while
    avoiding the repeated ancestor text a naive ``get_text`` would collect.
    """
    title = _help_title(soup, fallback_title)
    root = (
        soup.select_one("div.answerWrapper")
        or soup.select_one("div.qap761TextAnswer")
        or soup.select_one("div.layout-main")
    )
    if root is None:
        return []

    lines: list[str] = []
    _collect_help(root, lines)
    body = "\n".join(_help_dedupe(lines, title))
    if not body:
        return []
    return [Section(title=title or "Answer", body=body, level=2)]


def _help_title(soup: BeautifulSoup, fallback: str) -> str:
    """Return the article title, skipping the site's "Customer Support" banner."""
    heading = soup.find("h1", class_=lambda value: value and "qap761Title" in value)
    if heading is None:
        for candidate in soup.find_all("h1"):
            text = _clean(candidate.get_text(" ", strip=True))
            if text and text.lower() != "customer support":
                heading = candidate
                break
    if heading is None:
        return _clean(fallback)
    return _clean(heading.get_text(" ", strip=True)) or _clean(fallback)


def _collect_help(node: Tag, out: list[str]) -> None:
    """Collect leaf-block prose, numbering ordered-list steps in place."""
    if node.name == "table":
        out.extend(_table_lines(node))
        return

    children = [child for child in node.children if isinstance(child, Tag)]
    has_block_child = any(child.name in BLOCK_TAGS for child in children)

    if node.name in ("p", "li") or not has_block_child:
        text = _clean(node.get_text(" ", strip=True))
        if text:
            out.append(text)
        return

    for child in children:
        if child.attrs is None or child.name in STRIP_TAGS or _is_noise(child):
            continue
        if child.name == "table":
            out.extend(_table_lines(child))
        elif child.name in HEADING_TAGS:
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(f"### {text}")
        elif child.name == "ol":
            steps = child.find_all("li", recursive=False)
            if not steps:
                _collect_help(child, out)
                continue
            for index, step in enumerate(steps, start=1):
                text = _clean(step.get_text(" ", strip=True))
                if text:
                    out.append(f"{index}. {text}")
        elif child.name == "ul":
            for item in child.find_all("li", recursive=False):
                text = _clean(item.get_text(" ", strip=True))
                if text:
                    out.append(f"- {text}")
        elif child.name == "pre":
            continue
        else:
            _collect_help(child, out)


def _help_dedupe(lines: list[str], title: str) -> list[str]:
    """Drop article chrome, CTA lines, and repeated ancestor text."""
    cleaned: list[str] = []
    seen: set[str] = set()
    title_key = _clean(title).lower()

    for line in lines:
        text = _clean(line)
        if not text:
            continue
        lowered = text.lower()
        if lowered in HELP_NOISE_LINES:
            continue
        if any(lowered.startswith(prefix) for prefix in HELP_NOISE_PREFIXES):
            continue
        if any(fragment in lowered for fragment in HELP_DROP_FRAGMENTS):
            continue

        bare = re.sub(r"^(?:\d+\.|-)\s*", "", text).strip()
        if not bare:
            continue
        key = bare.lower()
        if key == title_key or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    return cleaned


def _scheme_sections(soup: BeautifulSoup, scheme: str) -> list[Section]:
    sections: list[Section] = []

    overview = _scheme_overview(soup, scheme)
    if overview:
        sections.append(Section(title="Overview", body=overview, level=2))

    fact_block = _by_class_prefix(soup, "fundDetails_fundDetailsContainer")
    if fact_block is not None:
        sections.append(
            Section(title="Key facts", body=_render_block(fact_block), level=2)
        )

    min_block = _by_class_prefix(soup, "minInvestments_tableContainer")
    if min_block is not None:
        sections.append(
            Section(
                title="Minimum investments",
                body=_render_block(min_block),
                level=2,
            )
        )

    exit_block = _by_class_prefix(soup, "exitLoadStampDutyTax_contentContainer")
    if exit_block is not None:
        # Keep only the exit-load row. The sibling rows carry tax/return figures
        # that must never become retrievable performance context.
        parts: list[str] = []
        for node in exit_block.select("[class*='exitLoadStampDutyTax_section']"):
            heading = node.find(["h3", "h4", "h5"])
            if heading is None:
                continue
            if "exit load" not in _clean(heading.get_text(" ", strip=True)).lower():
                continue
            parts.append(_render_block(node))
        body = "\n".join(part for part in parts if part) or _render_block(exit_block)
        body = re.sub(r"^###\s+exit load\s*\n", "", body, flags=re.IGNORECASE)
        sections.append(
            Section(title="Exit load", body=body, level=2)
        )

    about_block = _by_class_prefix(soup, "investmentObjective_container")
    if about_block is not None:
        sections.append(
            Section(title="About the fund", body=_render_block(about_block), level=2)
        )

    fund_house = _by_class_prefix(soup, "fundHouse_container")
    if fund_house is not None:
        blocks = fund_house.select("[class*='fundHouse_contentSection']")[:2]
        lines: list[str] = []
        for index, block in enumerate(blocks):
            rendered = _render_block(block).splitlines()
            if index == 0 and len(rendered) == 1 and rendered[0]:
                lines.append(f"AMC: {rendered[0]}")
            else:
                lines.extend(rendered)
        keep = ("AMC:", "Rank (total assets)", "Total AUM", "Date of Incorporation", "Launch Date")
        lines = [line for line in lines if line.startswith(keep)]
        sections.append(Section(title="Fund house", body="\n".join(lines), level=2))

    return sections


def _amc_sections(soup: BeautifulSoup) -> list[Section]:
    sections: list[Section] = []

    layout = soup.select_one("div.layout-main")
    hero = layout.find("section", recursive=False) if layout else None
    if hero is not None:
        sections.append(Section(title="Overview", body=_render_block(hero), level=2))

    heading = soup.find(
        lambda tag: tag.name == "h2" and "key information" in tag.get_text(" ", strip=True).lower()
    )
    table = heading.find_next("table") if heading is not None else None
    if table is not None:
        sections.append(
            Section(
                title="Key information",
                body="\n".join(_table_lines(table)),
                level=2,
            )
        )

    return sections


def _generic_sections(soup: BeautifulSoup) -> list[Section]:
    main = soup.select_one("main") or soup.body
    if main is None:
        return []
    return [Section(title="Content", body=_render_block(main), level=2)]


def _scheme_overview(soup: BeautifulSoup, fallback_title: str) -> str:
    header = soup.find("header")
    title = fallback_title
    pills: list[str] = []
    if header is not None:
        heading = header.find("h1")
        if heading is not None:
            title = _clean(heading.get_text(" ", strip=True)) or title
        for node in header.find_all(class_=True):
            classes = " ".join(node.get("class", []))
            if "pill" not in classes.lower():
                continue
            text = _clean(node.get_text(" ", strip=True))
            if text and text not in pills:
                pills.append(text)

    categories: list[str] = []
    risk = ""
    lock_in = ""
    for pill in pills:
        if LOCKIN_RE.search(pill):
            lock_in = pill
        elif RISK_RE.search(pill):
            risk = pill
        else:
            categories.append(pill)

    lines = [f"Scheme: {title}"]
    if categories:
        lines.append(f"Category: {' / '.join(categories)}")
    if risk:
        lines.append(f"Riskometer: {risk}")
    if lock_in:
        lines.append(f"Lock-in: {lock_in}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _strip_noise(soup: BeautifulSoup) -> None:
    # Decomposing a parent invalidates its children in the result list, so every
    # candidate is re-checked right before use.
    for tag in soup(STRIP_TAGS):
        if tag.attrs is not None:
            tag.decompose()
    for node in soup.find_all(class_=True):
        if node.attrs is None:
            continue
        if NOISE_CLASS_RE.search(" ".join(node.get("class", []))):
            node.decompose()
    for node in soup.find_all(attrs={"aria-haspopup": "dialog"}):
        if node.attrs is not None:
            node.decompose()


def _render_block(root: Tag) -> str:
    lines: list[str] = []
    _collect(root, lines)
    return "\n".join(_dedupe(lines))


def _collect(node: Tag, out: list[str]) -> None:
    children = [child for child in node.children if isinstance(child, Tag)]
    if not children:
        text = _clean(node.get_text(" ", strip=True))
        if text:
            out.append(text)
        return

    if node.name == "table":
        out.extend(_table_lines(node))
        return

    for child in children:
        if child.name in STRIP_TAGS or _is_noise(child):
            continue

        if child.name == "table":
            out.extend(_table_lines(child))
            continue

        if child.name in HEADING_TAGS:
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(f"### {text}")
            continue

        if child.name == "p":
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(text)
            continue

        if child.name == "li":
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(f"- {text}")
            continue

        pair = _as_pair(child)
        if pair is not None:
            label, value, label_is_heading = pair
            if label_is_heading:
                out.append(f"### {label}")
                out.append(value)
            elif ":" in label:
                out.append(f"{label} {value}")
            else:
                out.append(f"{label}: {value}")
            continue

        if _is_atomic(child):
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(text)
            continue

        _collect(child, out)


def _as_pair(node: Tag) -> tuple[str, str, bool] | None:
    """Return ``(label, value, label_is_heading)`` for a two-cell row."""
    if node.name not in ("div", "span", "li", "dl"):
        return None

    kids = [child for child in node.children if isinstance(child, Tag) and _text(child)]
    if len(kids) != 2:
        return None

    first, second = kids
    if not (_is_atomic(first) and _is_atomic(second)):
        return None
    # A label is one text block. ``benchmarkRow`` (two spans) is a row of its
    # own and must be rendered by recursing into it, not paired with its sibling.
    if len([child for child in first.find_all(True) if _text(child)]) > 1:
        return None

    label = _clean(first.get_text(" ", strip=True))
    value = _clean(second.get_text(" ", strip=True))
    if not label or not value or label == value:
        return None
    if len(label) > 90 or len(value) > 4000:
        return None
    if label.endswith(".") or label.endswith(","):
        return None
    if len(label) > 60 and ". " in label:
        return None
    if len(label) > 200 and len(value) > 200:
        return None
    return label, value, first.name in HEADING_TAGS


def _is_atomic(node: Tag) -> bool:
    if node.name in HEADING_TAGS or node.name == "p":
        return True
    for descendant in node.find_all(BLOCK_TAGS):
        if descendant is node:
            continue
        if _text(descendant):
            return False
    return True


def _table_lines(table: Tag) -> list[str]:
    lines: list[str] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        values = [_clean(cell.get_text(" ", strip=True)) for cell in cells]
        values = [value for value in values if value]
        if not values:
            continue
        if len(values) == 2 and not row.find_all("th"):
            lines.append(f"{values[0]}: {values[1]}")
        else:
            lines.append("| " + " | ".join(values) + " |")
    return lines


def _dedupe(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        text = _clean(line)
        if not text or text.lower() in UI_NOISE_LINES:
            continue
        if NOISE_LINE_RE.match(text):
            continue
        if cleaned and cleaned[-1] == text:
            continue
        cleaned.append(text)
    return cleaned


def _text(node: Tag) -> str:
    return _clean(node.get_text(" ", strip=True))


def _clean(value: str) -> str:
    value = value.replace("\xa0", " ").replace("\u200b", "")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r"\.\s*;$", ".", value)
    return value.strip()


def _is_noise(node: Tag) -> bool:
    if node.name in STRIP_TAGS:
        return True
    return NOISE_CLASS_RE.search(" ".join(node.get("class", []))) is not None


def _by_class_prefix(soup: BeautifulSoup, prefix: str) -> Tag | None:
    for node in soup.find_all(True, class_=True):
        if any(str(name).startswith(prefix) for name in node.get("class", [])):
            return node
    return None


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _parse_entry(
    settings: Settings,
    entry: dict[str, Any],
    source: SourceRow,
    raw_dir: Path,
    out_dir: Path,
) -> PageReport:
    slug = entry.get("slug") or slug_for(entry["url"])
    html_path = raw_dir / f"{slug}.html"
    page_type = source.page_type

    report = PageReport(
        id=source.id,
        slug=slug,
        url=source.url,
        scheme=source.scheme,
        page_type=page_type,
        ok=False,
        char_len=0,
        section_count=0,
    )

    if not entry.get("ok", False):
        report.reason = f"fetch_failed: {entry.get('error') or 'unknown'}"
        return report

    if not html_path.exists():
        report.reason = f"raw_missing: {html_path.name}"
        return report

    html = html_path.read_text(encoding="utf-8", errors="replace")
    doc = parse_html(
        html,
        scheme=source.scheme,
        source_url=source.url,
        source_title=source.scheme,
        page_type=page_type,
        fetched_at=entry.get("fetched_at") or "",
        raw_sha256=entry.get("sha256") or "",
    )

    hits = fact_hits(doc)
    required = list(REQUIRED_FACTS.get(page_type, ()))
    if page_type == "scheme" and any(key in slug for key in LOCKIN_SCHEMES):
        required.append("lock_in")
    missing = [name for name in required if not hits.get(name, {}).get("found")]

    report.char_len = doc.char_len
    report.section_count = len(doc.sections)
    report.facts = hits
    report.missing = missing

    if doc.char_len < settings.corpus.min_clean_chars:
        report.reason = (
            f"too_short: {doc.char_len} < corpus.min_clean_chars "
            f"({settings.corpus.min_clean_chars})"
        )
    elif not doc.sections:
        report.reason = "no_sections"
    elif missing:
        report.reason = "missing_facts: " + ", ".join(missing)
    else:
        report.ok = True

    (out_dir / f"{slug}.md").write_text(doc_to_markdown(doc), encoding="utf-8")
    (out_dir / f"{slug}.json").write_text(
        json.dumps(asdict(doc), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _summary(reports: list[PageReport]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for report in reports:
        by_type[report.page_type] = by_type.get(report.page_type, 0) + 1
    return {
        "pages": len(reports),
        "ok": sum(1 for report in reports if report.ok),
        "quarantined": sum(1 for report in reports if not report.ok),
        "by_page_type": by_type,
        "total_chars": sum(report.char_len for report in reports),
        "gaps": sum(len(report.missing) for report in reports),
    }


__all__ = [
    "FACT_PROBES",
    "PageReport",
    "doc_to_markdown",
    "fact_hits",
    "parse_all",
    "parse_html",
]
