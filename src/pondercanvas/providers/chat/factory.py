from google.adk.models import BaseLlm, Gemini
from google.adk.models.lite_llm import LiteLlm

from pondercanvas.config.settings import EffectiveSettings
from pondercanvas.providers._gemini import gemini_http_options

_NON_GEMINI_KEY_FIELD = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}


def build_chat_model(settings: EffectiveSettings) -> BaseLlm:
    """Rebuilt fresh per pipeline run (never cached globally) so runtime
    provider/key swaps from the Gradio settings panel are trivially correct
    without a restart, and without mutating os.environ."""
    if settings.chat_provider == "gemini":
        return Gemini(
            model=settings.chat_model_id,
            client_kwargs={
                "api_key": settings.google_api_key,
                "http_options": gemini_http_options(),
            },
        )

    key_field = _NON_GEMINI_KEY_FIELD.get(settings.chat_provider)
    if key_field is None:
        available = ", ".join(["gemini", *_NON_GEMINI_KEY_FIELD])
        raise ValueError(
            f"Unknown chat provider {settings.chat_provider!r}; available: {available}"
        )

    api_key = getattr(settings, key_field)
    return LiteLlm(model=f"{settings.chat_provider}/{settings.chat_model_id}", api_key=api_key)
