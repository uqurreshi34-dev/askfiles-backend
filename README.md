# AskFiles Backend

Django REST API backend for AskFiles Android app.

## Stack
- Django REST Framework
- Groq LLM API (llama-3.3-70b-versatile)
- Render (auto-deploy on git push)

## Endpoints
- `/api/search/` — AI file search via Groq
- `/api/auth/` — User authentication

## Deploy
Auto-deploys to Render on push to main.
