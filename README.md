# Groww Help FAQ Assistant

A facts-only RAG assistant for **Groww platform help**. It answers account, KYC,
charges, order, demat, and payment questions from five official `groww.in` help
articles, with exactly one source link and a freshness line.

This is a sibling rebuild of the earlier HDFC mutual-fund demo. The pipeline,
UI, inspector, and guardrails are reused; the corpus and wording are Groww help
only. There is **no HDFC content and no mutual-fund scheme content**.

## Scope

| # | Official page | Used for |
|---|---|---|
| 1 | [Do you charge for processing KYC?](https://groww.in/help/my-account/ma-kyc/do-you-charge-for-processing-kyc) | KYC processing fees |
| 2 | [How to change password on Groww?](https://groww.in/help/my-account/ma-account-details/how-to-change-password-on-groww) | password-change steps |
| 3 | [What are account opening charges?](https://groww.in/help/stocks/sx-pricing/what-are-account-opening-charges) | demat account opening charges |
| 4 | [When will the stocks be delivered to my demat account?](https://groww.in/help/stocks/order/when-will-the-stocks-be-delivered-to-my-demat-account-2) | stock delivery timeline |
| 5 | [Why is my 'Available to Invest' not the same as my 'Available to Withdraw'?](https://groww.in/help/payments/payments/why-is-my-available-to-invest-not-the-same-as-my-available-to-withdraw-1) | available balance and withdrawal timing |

Anything outside those pages gets a transparent "not in my scope" answer, never
a guess and never a different source.

## Guardrails

- Answers only from the five saved official snapshots; no live browsing at
  question time.
- At most three sentences, including exactly one citation.
- The freshness line is generated from chunk metadata, not the model.
- Refuses investment/trading advice and return or performance claims.
- Refuses to accept PAN, Aadhaar, account numbers, OTPs, emails, or phone
  numbers. PII is scanned before retrieval, so a hit never reaches the vector
  store or a provider.
- How-to questions are in scope, because the corpus is procedural help content.
- If retrieval cannot clear its score gate, no LLM call is made.

## Pipeline

```
sources.csv
  -> fetch       data/01_raw/          saved HTML + manifest
  -> parse       data/02_clean/        clean documents + parse report
  -> load        data/02_clean/        validated documents JSONL
  -> chunk       data/03_chunks/       6 section windows, 0 fact atoms
  -> embed       data/04_embeddings/   MiniLM vectors
  -> index       data/05_chroma/       Chroma collection groww_help_faq_v1
```

At question time:

```
guard PII -> resolve topic -> retrieve -> score gate -> prompt -> LLM
          -> validate -> repair once -> extractive fallback -> cite
```

Every retrieval and answer is persisted under `data/06_retrieval/` and
`data/07_answers/` for developer inspection. The inspector is not rendered in
the visitor UI.

## Setup

Python 3.12. The project was built against the existing sibling environment, so
you can reuse it without reinstalling torch:

```powershell
..\groww-mf-faq-assistant\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or create a local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set one provider key. The app also runs fully
offline with extractive answers when no key is configured.

## Build the corpus

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/check_sources.py
python -m ingest.run all
```

To rebuild after editing parsing or chunking only:

```powershell
python -m ingest.run chunk embed index
```

Current corpus: 5 documents, 2,393 clean characters, 6 section-window chunks,
384-dimensional MiniLM embeddings.

## Run

```powershell
python -m streamlit run app/streamlit_app.py
```

or:

```powershell
.\run.ps1
```

Open the local Streamlit URL printed in the terminal. The sidebar has the scope
and sources notes. Stage artifacts remain on disk for development and are not
shown to visitors.

Example questions:

- Do you charge for KYC verification?
- How do I change my Groww password?
- When will my stocks reach my demat account?

## Verify

```powershell
python -m pytest -q
python -m ruff check .
```

Expected refusal paths:

| Question | Result |
|---|---|
| `What is the expense ratio of HDFC Mid Cap Fund?` | `not_found` / out of scope |
| `Should I buy HDFC Bank shares?` | refused / advice |
| `My PAN is ABCDE1234F` | refused / PII, value not stored |
| `What is the weather today?` | `not_found` / below global score gate |

## Layout

```
app/        Streamlit UI, source cards, developer-only inspector
scripts/    optional local FastAPI surface (requirements-api.txt) + offline tools
config/     config.yaml + typed settings
core/       guard, intent, resolve, retrieval, prompt, validate, answer
ingest/     fetch, parse, load, chunk, embed, index
prompts/    versioned system prompt
sources/    sources.csv + SOURCES.md scope record
tests/      focused parse, resolve, intent, chunk, fallback, validate tests
data/       generated artifacts (raw pages and traces stay local; the small runtime index is committed)
```

## Limitations

- Only five pages are indexed; a missing answer is expected and reported.
- The corpus snapshot is dated 2026-11-06 and can lag the live help centre.
- The demo is educational, not customer support, and gives no advice.

See `sources/SOURCES.md` for the source-selection and exclusion record.
