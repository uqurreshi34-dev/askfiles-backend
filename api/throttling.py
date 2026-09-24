"""Server-side limits on the AI endpoints.

The app already limits itself, but anything the app enforces can be skipped by
calling the API directly, and the API key inside the app can be read out of the
APK. So the server keeps its own count, in two layers:

- Per client, checked before anything else: a short burst limit and an hourly
  and daily ceiling, each far above what a real person asking questions in the
  app can reach. Every attempt counts, including ones with a wrong key, so
  guessing is throttled too.
- Global, checked only once a request is authorised and valid, just before the
  model is called: one hourly and daily ceiling across everyone. This is the
  cost cap. It holds even if a caller finds a way to look like many clients,
  and rejected requests never use it up.

Every rate lives in settings and comes from an environment variable, so they
can be tuned on Render without a code change.
"""

from ipaddress import ip_address

from rest_framework.throttling import BaseThrottle, SimpleRateThrottle


# Set by Cloudflare, which sits in front of every Render service and overwrites any value a
# caller sends. X-Forwarded-For is not used: Render keeps whatever the caller put in it and
# adds to the list, so its first entry is whatever the caller chose.
_CLIENT_HEADERS = ("HTTP_CF_CONNECTING_IP", "HTTP_TRUE_CLIENT_IP")


def client_ip(request):
    """The address of whoever sent the request.

    Checked against Render's real headers: a request sent with a made-up X-Forwarded-For
    still arrived with CF-Connecting-IP set to the sender's true address. Anything that is
    not a valid IP address falls back to REMOTE_ADDR rather than becoming a key a caller can
    choose freely.
    """
    for header in _CLIENT_HEADERS:
        value = request.META.get(header, "").strip()

        if value:
            try:
                return str(ip_address(value))
            except ValueError:
                pass

    return request.META.get("REMOTE_ADDR", "") or "unknown"


class _Layer(SimpleRateThrottle):
    """One named rate from settings, keyed by client or shared by all."""

    def __init__(self, scope, shared):
        self.scope = scope
        self.shared = shared
        super().__init__()

    def get_cache_key(self, request, view):
        ident = "all" if self.shared else client_ip(request)
        return self.cache_format % {"scope": self.scope, "ident": ident}


def _first_refusal(layers, request, view):
    """Check layers in order and stop at the first one that says no.

    Stopping matters: a layer that is never asked records nothing, so a
    request refused early does not use up the layers after it.
    """
    for layer in layers:
        if not layer.allow_request(request, view):
            return layer.wait()

    return None


class ClientThrottle(BaseThrottle):
    """Per-client burst, hourly and daily limits, as one DRF throttle."""

    scopes = ("ai_client_burst", "ai_client_hour", "ai_client_day")

    def allow_request(self, request, view):
        layers = [_Layer(scope, shared=False) for scope in self.scopes]
        self._wait = _first_refusal(layers, request, view)
        return self._wait is None

    def wait(self):
        return getattr(self, "_wait", None)


GLOBAL_SCOPES = ("ai_global_hour", "ai_global_day")


def global_wait(request, view=None):
    """Spend one unit of the shared budget. Seconds to wait if it is gone."""
    layers = [_Layer(scope, shared=True) for scope in GLOBAL_SCOPES]
    return _first_refusal(layers, request, view)

