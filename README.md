# AskFiles Backend

Django REST API backend for AskFiles Android app.

## Stack
- Django REST Framework
- Existing AskFiles AI via Groq
- JARVIS Mobile via Claude + Edge TTS
- Render (auto-deploy on git push)

## Endpoints
- `/api/ask-ai/` — existing AskFiles AI via Groq
- `/api/health/` — backend health check
- `/api/jarvis/organise/` — JARVIS folder-organisation planning
- `/api/jarvis/audio/` — JARVIS neural voice audio

## JARVIS environment
Set these in Render (and in the local `.env` for development):

- `ANTHROPIC_API_KEY` — Anthropic API key used only by the backend
- `JARVIS_SERVICE_KEY` — client key sent by AskFiles to the JARVIS endpoints
- `JARVIS_MODEL` — optional, defaults to `claude-opus-5`
- `JARVIS_MAX_TOKENS` — optional, defaults to `3072`
- `JARVIS_EFFORT` — optional, defaults to `low`
- `JARVIS_VOICE` — optional, defaults to `en-GB-RyanNeural`
- `JARVIS_VOICE_RATE` — optional, defaults to `-7%`
- `JARVIS_VOICE_PITCH` — optional, defaults to `-4Hz`

The JARVIS organiser returns the same `summary`, `moves`, and `create_folders` shape used by AskFiles. The backend never receives or changes the user's actual files.

## Deploy
Auto-deploys to Render on push to main.
