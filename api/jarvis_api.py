import os
import secrets

from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .jarvis_service import JarvisServiceError, audio_bytes, organise


MAX_TTS_CHARS = 2000


def _check_access(request):
    expected = (os.getenv("JARVIS_SERVICE_KEY") or "").strip()
    supplied = (request.headers.get("X-JARVIS-Key") or "").strip()

    if not expected:
        return Response(
            {"error": "JARVIS Mobile is not configured on the server."},
            status=503,
        )

    if not secrets.compare_digest(supplied, expected):
        return Response({"error": "Unauthorized."}, status=401)

    return None


@api_view(["POST"])
@permission_classes([AllowAny])
def jarvis_organise(request):
    access_error = _check_access(request)
    if access_error is not None:
        return access_error

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


@api_view(["POST"])
@permission_classes([AllowAny])
def jarvis_audio(request):
    access_error = _check_access(request)
    if access_error is not None:
        return access_error

    payload = request.data if isinstance(request.data, dict) else {}
    text = str(payload.get("text") or "").strip()

    if not text:
        return Response({"error": "Nothing to say."}, status=400)

    if len(text) > MAX_TTS_CHARS:
        return Response({"error": "That reply is too long to speak."}, status=400)

    try:
        data = audio_bytes(text)
    except JarvisServiceError as error:
        return Response({"error": str(error)}, status=503)
    except Exception as error:
        print(f"[JARVIS Mobile] audio endpoint failed: {error}")
        return Response({"error": "JARVIS voice is unavailable right now."}, status=503)

    if not data:
        return Response({"error": "No audio."}, status=503)

    return HttpResponse(
        data,
        content_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )
