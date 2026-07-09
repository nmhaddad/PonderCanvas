from google.genai import types

from pondercanvas.providers._gemini import gemini_http_options


class TestGeminiHttpOptions:
    def test_returns_http_options_with_retry(self):
        options = gemini_http_options()
        assert isinstance(options, types.HttpOptions)
        assert options.retry_options is not None

    def test_retries_on_rate_limit_and_server_errors(self):
        retry = gemini_http_options().retry_options
        assert 429 in retry.http_status_codes  # RESOURCE_EXHAUSTED
        assert 503 in retry.http_status_codes  # UNAVAILABLE

    def test_uses_exponential_backoff_across_multiple_attempts(self):
        retry = gemini_http_options().retry_options
        assert retry.attempts > 1
        assert retry.exp_base > 1.0
