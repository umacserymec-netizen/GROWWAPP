# Deploying on Vercel

This app runs as a **Container** on Vercel, not as a FastAPI project. The
repository root contains `Dockerfile.vercel`; Vercel's framework detector gives
that file priority over Python/FastAPI detection, and `vercel.json` pins the
preset as well.

## Required project settings

| Setting | Value |
|---|---|
| Application Preset / Framework Preset | **Container** |
| Root Directory | `./` |
| Build Command | *(empty / None)* |
| Output Directory | *(empty / None)* |
| Install Command | *(empty / None)* |

If an existing Vercel project was created with the **FastAPI** preset, change it
under **Settings → Build and Deployment → Framework Preset → Container**, then
redeploy. Deleting the project and importing the repository again also works.

## Environment variables

The app works with no environment variables: answers fall back to the
extractive, source-grounded path. To enable LLM-written answers, add:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq API key |

Never commit `.env` or `.env.local`.

## Why a container

Streamlit needs a long-lived HTTP/WebSocket server, which Vercel's serverless
Python runtime does not provide. The Container runtime runs the exact same
Streamlit process used locally.

At boot the container copies the committed index (`data/`) and the Hugging Face
model cache to `/tmp`, because Vercel's image filesystem is read-only. The app
then listens on `$PORT` (default `80`).

## Local run

```powershell
.\run.ps1
```

The optional local FastAPI surface is not needed on Vercel. Install its extra
dependencies only when running `scripts/api.py`:

```powershell
pip install -r requirements.txt -r scripts/requirements-api.txt
```
