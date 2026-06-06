import os
import requests as http_requests
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response


@api_view(['GET', 'HEAD'])
@permission_classes([AllowAny])
def health(request):
    return Response({'status': 'ok'})


API_KEY = os.getenv('ASKFILES_API_KEY')
WORKER_URL = 'https://groq-proxy.u-qurreshi34.workers.dev'


@api_view(['POST'])
@permission_classes([AllowAny])
def ask_ai(request):
    if API_KEY and request.headers.get('X-API-Key') != API_KEY:
        return Response({'error': 'Unauthorized'}, status=401)
    question = request.data.get('question', '').strip()
    context = request.data.get('context', '').strip()

    if not question:
        return Response({'error': 'Question is required'}, status=400)

    try:
        worker_response = http_requests.post(
            WORKER_URL,
            json={
                'model': 'llama-3.3-70b-versatile',
                'max_tokens': 4096,
                'temperature': 0.3,
                'messages': [
                    {
                        'role': 'system',
                        'content': f"""You are AskFiles AI, a helpful file manager assistant built into the AskFiles app.
The user's device file context is below. Read it carefully before answering.

{context}

Rules:
- Never confuse image filenames with video filenames — they are listed separately.
- For downloads, the largest files by name and size are provided. Use them to answer questions about large downloads accurately.
- Always use the folder location provided in brackets (e.g. "in Downloads", "in My Files", "in Camera Roll") when stating where a file is located. Never guess a file's location based on its type.
- APK files are Android app installer files. Treat them like any other file — report their name, size and location accurately.
- When answering "what's my largest file", always use the "Top 10 largest files across all storage" list provided in the context — not the per-category lists.
- Screenshots are files whose names start with "Screenshot_". Use the screenshotCount field for the exact number, do not count manually.
- Image and video format breakdowns are from a sample only, not all files. Never state format totals as absolute counts — only use the total image/video count from File counts for totals.
- PNG, JPG, JPEG, HEIC, GIF, WEBP are ALL image formats. Never add notes like "(this is actually a jpg)" or "(included as it is an image)" — jpg IS an image, treat it as such with zero comment.
- MP4, MKV, AVI, MOV, WEBM are ALL video formats. Never add notes like "(this is actually a video)" — treat them as videos with zero comment. Never call a video an image.
- Keep answers short and practical — 3-4 sentences max unless a list is genuinely needed.
- Do not make up files that aren't in the context.
- Never use markdown formatting. No asterisks, no bold, no bullet points with *. If you need a list use plain numbered lines like "1. filename" or plain sentences.
- 'Other' storage represents system and app data the user cannot access or manage. Never mention it when answering questions about largest files or folders.
- You ONLY answer questions about the user's files and storage on their device. If the user asks about anything else — sports, news, general knowledge, weather, people, places — respond with exactly: "I can only help with questions about your files and storage. Try asking about your largest files, storage usage, or what's on your device." Do not answer off-topic questions under any circumstances."""
                    },
                    {
                        'role': 'user',
                        'content': question
                    }
                ]
            },
            timeout=30
        )
        result = worker_response.json()
        answer = result['choices'][0]['message']['content']
        return Response({'answer': answer})

    except Exception as e:
        print(f'Groq error: {e}')
        return Response({'error': 'AI unavailable. Try again.'}, status=500)
