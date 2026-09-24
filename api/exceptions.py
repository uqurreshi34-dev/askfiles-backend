"""One error shape for every response the app can receive.

The app reads the 'error' field. DRF's own refusals (too many requests, a
malformed body) only fill in 'detail', which the app would show as an empty
answer. This adds 'error' alongside it, in words a person can act on.
"""

from rest_framework.exceptions import Throttled
from rest_framework.views import exception_handler


def api_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if response is None or not isinstance(response.data, dict):
        return response

    if isinstance(exc, Throttled):
        response.data['error'] = 'Too many questions in a short time. Try again in a few minutes.'
    else:
        detail = response.data.get('detail')
        response.data.setdefault('error', str(detail) if detail else 'Request not accepted.')

    return response
