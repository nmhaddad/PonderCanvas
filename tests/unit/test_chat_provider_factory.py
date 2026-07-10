import pytest
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm

from pondercanvas.config.settings import AppSettings, resolve_settings
from pondercanvas.providers.chat.factory import build_chat_model


def _effective(**overrides):
    return resolve_settings(AppSettings(_env_file=None, **overrides))  # type: ignore[call-arg]


class TestBuildChatModel:
    def test_gemini_provider_returns_gemini_instance(self):
        settings = _effective(chat_provider="gemini", chat_model_id="gemini-2.5-flash")
        model = build_chat_model(settings)
        assert isinstance(model, Gemini)
        assert model.model == "gemini-2.5-flash"

    def test_gemini_provider_always_wires_enterprise_via_client_kwargs(self):
        # Always ADC/Enterprise mode now, no API key setting at all for this
        # provider.
        settings = _effective(chat_provider="gemini")
        model = build_chat_model(settings)
        assert isinstance(model, Gemini)
        assert model.client_kwargs["enterprise"] is True
        assert "api_key" not in model.client_kwargs

    def test_gemini_provider_enables_retry_on_rate_limit(self):
        # Gemini answers 429 RESOURCE_EXHAUSTED on a rate/quota limit; the
        # client only retries when http_options.retry_options is supplied.
        settings = _effective(chat_provider="gemini")
        model = build_chat_model(settings)
        assert isinstance(model, Gemini)
        retry = model.client_kwargs["http_options"].retry_options
        assert 429 in retry.http_status_codes
        assert retry.attempts > 1

    def test_openai_provider_returns_lite_llm_with_prefixed_model(self):
        settings = _effective(
            chat_provider="openai", chat_model_id="gpt-5", openai_api_key="o-secret"
        )
        model = build_chat_model(settings)
        assert isinstance(model, LiteLlm)
        assert model.model == "openai/gpt-5"

    def test_openai_provider_forwards_api_key_to_litellm(self):
        settings = _effective(chat_provider="openai", openai_api_key="o-secret")
        model = build_chat_model(settings)
        assert model._additional_args["api_key"] == "o-secret"

    def test_anthropic_provider_returns_lite_llm_with_prefixed_model(self):
        settings = _effective(
            chat_provider="anthropic", chat_model_id="claude-x", anthropic_api_key="a-secret"
        )
        model = build_chat_model(settings)
        assert isinstance(model, LiteLlm)
        assert model.model == "anthropic/claude-x"
        assert model._additional_args["api_key"] == "a-secret"

    def test_unknown_provider_raises_value_error(self):
        settings = _effective(chat_provider="not-a-real-provider")
        with pytest.raises(ValueError, match="Unknown chat provider"):
            build_chat_model(settings)

    def test_fresh_instance_built_each_call_no_caching(self):
        settings = _effective(chat_provider="gemini")
        first = build_chat_model(settings)
        second = build_chat_model(settings)
        assert first is not second
