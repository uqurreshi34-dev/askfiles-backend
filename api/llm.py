"""Ask AI's model, with failover, the same order as JARVIS.

1. Groq, through the Cloudflare Worker at WORKER_URL (the Worker holds the
   Groq key, so it never sits on Render).
2. Gemini, Google's OpenAI-compatible endpoint (GEMINI_API_KEY).
3. DeepSeek, paid per use, the last resort (DEEPSEEK_API_KEY).

A provider with no URL or key set is skipped. A provider that reports it is
rate limited or out of credit rests for 15 minutes, so every request does not
pay for a failed call first. Each provider gets a short timeout and the whole
chain stays inside gunicorn's 30-second request limit.
"""

import os
import time

import requests

# After a quota or credit refusal, skip that provider for this long.
_COOLDOWN_SECONDS = 15 * 60
# Connect, read. Groq answers in a second or two; the others a few seconds.
_TIMEOUT = (4, 18)
# Gunicorn kills a request at 30 s. Stop trying new providers before that.
_DEADLINE_SECONDS = 26
# Not worth starting a provider with less time than this left.
_MIN_SECONDS_LEFT = 6

_resting_until = {}


def _providers():
    return [
        {
            'name': 'groq',
            'url': (os.getenv('WORKER_URL') or '').strip(),
            'key': None,  # the Worker adds it
            'model': (os.getenv('GROQ_MODEL') or 'openai/gpt-oss-20b').strip(),
        },
        {
            'name': 'gemini',
            'url': 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
            'key': (os.getenv('GEMINI_API_KEY') or '').strip(),
            'model': (os.getenv('GEMINI_MODEL') or 'gemini-3.6-flash').strip(),
            'extra': {'reasoning_effort': 'low'},
        },
        {
            'name': 'deepseek',
            'url': 'https://api.deepseek.com/chat/completions',
            'key': (os.getenv('DEEPSEEK_API_KEY') or '').strip(),
            'model': (os.getenv('DEEPSEEK_MODEL') or 'deepseek-flash').strip(),
            # Thinking costs seconds and tokens; these answers are short.
            'extra': {'thinking': {'type': 'disabled'}},
        },
    ]


class NoModelAvailable(Exception):
    """Every provider was skipped, rested or failed."""


def _quota_refusal(status, text):
    text = text.lower()
    return status in (402, 429) or 'quota' in text or 'rate limit' in text or 'insufficient balance' in text


def _post(provider, body, timeout):
    headers = {'Authorization': f"Bearer {provider['key']}"} if provider['key'] else {}
    return requests.post(provider['url'], json=body, headers=headers, timeout=timeout)


def complete(messages, max_tokens=4096, temperature=0.3):
    """The first provider's reply text, and which provider gave it."""
    started = time.monotonic()
    tried = []

    for provider in _providers():
        name = provider['name']

        if not provider['url'] or (name != 'groq' and not provider['key']):
            continue
        if _resting_until.get(name, 0) > time.time():
            tried.append(f'{name}:resting')
            continue

        left = _DEADLINE_SECONDS - (time.monotonic() - started)
        if left < _MIN_SECONDS_LEFT:
            tried.append(f'{name}:no time')
            break
        timeout = (_TIMEOUT[0], min(_TIMEOUT[1], left - _TIMEOUT[0]))

        body = {
            'model': provider['model'],
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            **provider.get('extra', {}),
        }

        try:
            response = _post(provider, body, timeout)

            if response.status_code == 400 and 'reasoning_effort' in body:
                # Only a speed-up; a model that refuses it is asked again without.
                body.pop('reasoning_effort')
                response = _post(provider, body, timeout)

            if response.status_code != 200:
                text = response.text[:300]
                if _quota_refusal(response.status_code, text):
                    _resting_until[name] = time.time() + _COOLDOWN_SECONDS
                print(f'ask-ai {name} answered {response.status_code}: {text}')
                tried.append(f'{name}:{response.status_code}')
                continue

            result = response.json()
            choices = result.get('choices') or []
            answer = ((choices[0].get('message') or {}).get('content') or '').strip() if choices else ''
            if not answer:
                print(f'ask-ai {name} returned no answer: {str(result)[:300]}')
                tried.append(f'{name}:empty')
                continue

            if tried:
                print(f"ask-ai failover {' > '.join(tried)} > {name}")
            return answer, name

        except Exception as error:  # timeout, connection, bad JSON
            print(f'ask-ai {name} failed: {type(error).__name__}: {str(error)[:200]}')
            tried.append(f'{name}:{type(error).__name__}')

    raise NoModelAvailable(' > '.join(tried) or 'no provider configured')
