import os

from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from .jarvis_service import JarvisServiceError, organise


MAX_TTS_CHARS = 2000
GOOGLE_WEB_CLIENT_ID = (os.getenv("GOOGLE_WEB_CLIENT_ID") or "").strip()


def _authenticated_google_user(request):
    if not GOOGLE_WEB_CLIENT_ID:
        return None, Response(
            {"error": "JARVIS authentication is not configured."},
            status=503,
        )

    header = (request.headers.get("Authorization") or "").strip()

    if not header.startswith("Bearer "):
        return None, Response(
            {"error": "Authentication required."},
            status=401,
        )

    token = header[7:].strip()

    if not token:
        return None, Response(
            {"error": "Authentication required."},
            status=401,
        )

    try:
        claims = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            audience=GOOGLE_WEB_CLIENT_ID,
        )
    except Exception:
        return None, Response(
            {"error": "Invalid or expired authentication token."},
            status=401,
        )

    subject = str(claims.get("sub") or "").strip()

    if not subject:
        return None, Response(
            {"error": "Invalid authentication token."},
            status=401,
        )

    return claims, None


@api_view(["POST"])
@permission_classes([AllowAny])
def jarvis_organise(request):
    _, auth_error = _authenticated_google_user(request)

    if auth_error is not None:
        return auth_error

    payload = request.data if isinstance(request.data, dict) else {}
    current_path = str(payload.get("current_path") or "").strip()
    current_folder = str(payload.get("current_folder") or "").strip()
    items = payload.get("items")
    existing_child_folders = payload.get("existing_child_folders")

    if not current_path or not isinstance(items, list):
        return Response({"error": "Invalid AskFiles folder context."}, status=400)

    if not isinstance(existing_child_folders, list):
        existing_child_folders = []

    try:
        plan = organise(
            current_path=current_path,
            current_folder=current_folder or "Current folder",
            items=items,
            existing_child_folders=existing_child_folders,
        )
    except JarvisServiceError as error:
        return Response({"error": str(error)}, status=503)
    except Exception as error:
        print(f"[JARVIS Mobile] organisation endpoint failed: {error}")
        return Response(
            {"error": "I couldn't plan that folder right now, sir."},
            status=500,
        )

    return Response(plan)
