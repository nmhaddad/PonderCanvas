from google.genai import types

# Gemini rejects calls with 429 RESOURCE_EXHAUSTED when a rate limit or quota
# is hit, and 5xx on transient server hiccups. The google-genai client only
# retries when retry_options is passed -- otherwise the very first 429 raises.
# A per-minute rate cap (the common case) clears within seconds, so exponential
# backoff recovers it transparently; a hard daily-quota exhaustion still
# surfaces once the attempts are spent.
_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5,  # includes the original request
    initial_delay=1.0,  # seconds
    max_delay=30.0,
    exp_base=2.0,
    jitter=1.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)


def gemini_http_options() -> types.HttpOptions:
    """Shared genai.Client HTTP options: retry transient rate-limit (429) and
    server (5xx) errors with exponential backoff. Applied to every Gemini
    client this app constructs (extraction/evaluation, image, grounding)."""
    return types.HttpOptions(retry_options=_RETRY_OPTIONS)
