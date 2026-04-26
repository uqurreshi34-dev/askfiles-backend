import os
from groq import Groq
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response


@api_view(['GET'])
@permission_classes([AllowAny])
def health(request):
    return Response({'status': 'ok'})


@api_view(['POST'])
@permission_classes([AllowAny])
def ask_ai(request):
    question = request.data.get('question', '').strip()
    context = request.data.get('context', '').strip()

    if not question:
        return Response({'error': 'Question is required'}, status=400)

    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return Response({'error': 'AI not configured'}, status=500)

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            max_tokens=4096,
            temperature=0.3,
            messages=[
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
- PNG, JPG, JPEG, HEIC, GIF, WEBP are ALL image formats. Never add notes like "(this is actually a jpg)" or "(included as it is an image)" — jpg IS an image, treat it as such with zero comment.
- MP4, MKV, AVI, MOV, WEBM are ALL video formats. Never add notes like "(this is actually a video)" — treat them as videos with zero comment. Never call a video an image.
- Keep answers short and practical — 3-4 sentences max unless a list is genuinely needed.
- Do not make up files that aren't in the context.
- Never use markdown formatting. No asterisks, no bold, no bullet points with *. If you need a list use plain numbered lines like "1. filename" or plain sentences.
- Do not recount files from the filename list — always use the exact counts provided in the context above. But DO list filenames when the user asks to see them.
- If the user has 50 or fewer files, list ALL of them without any preamble like "here are the first X". Just say "You have X images:" and list them all.
- Only if the user has more than 50 files, count exactly how many filenames you are about to list, then say "Here are X of your Y total" where X is the exact number you are listing, then list them.
- 'Other' storage represents system and app data the user cannot access or manage. Never mention it when answering questions about largest files or folders.
- You ONLY answer questions about the user's files and storage on their device. If the user asks about anything else — sports, news, general knowledge, weather, people, places — respond with exactly: "I can only help with questions about your files and storage. Try asking about your largest files, storage usage, or what's on your device." Do not answer off-topic questions under any circumstances."""
                },
                {
                    'role': 'user',
                    'content': question
                }
            ]
        )
        answer = completion.choices[0].message.content
        return Response({'answer': answer})

    except Exception as e:
        print(f'Groq error: {e}')
        return Response({'error': 'AI unavailable. Try again.'}, status=500)
