import gradio as gr

from pondercanvas.config.constants import (
    CHAT_PROVIDERS,
    IMAGE_PROVIDERS,
    MAX_ITERATIONS_CAP,
    REFINEMENT_MODES,
)
from pondercanvas.config.settings import AppSettings, RuntimeSettingsOverlay

# Order here must match the positional argument order of fields_to_overlay
# and the order the fields list is spread into event handler inputs in
# ui/app.py.
SETTINGS_FIELD_ORDER = (
    "chat_provider",
    "chat_model_id",
    "image_provider",
    "image_model_id",
    "google_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "stability_api_key",
    "unsplash_api_key",
    "gemini_image_api_key",
    "gemini_image_enterprise",
    "gemini_image_search_enabled",
    "refinement_mode",
    "max_iterations",
    "eval_pass_threshold",
    "siglip_enabled",
    "siglip_weight",
)


def build_settings_panel() -> list[gr.components.Component]:
    # Non-secret fields are seeded from the actual resolved env/.env defaults
    # (not hardcoded constants) so the panel reflects what's really
    # configured on page load -- this matters especially for checkboxes:
    # since they always submit a definite True/False (never "unset"), an
    # unchanged checkbox permanently overrides an env-configured value
    # otherwise (see RuntimeSettingsOverlay/resolve_settings precedence).
    # API key fields are deliberately left blank regardless of .env, so
    # secrets never get written into page HTML.
    defaults = AppSettings()

    with gr.Accordion("Settings", open=False):
        gr.Markdown(
            "A Google API key is always required, for brief extraction, evaluation, "
            "and Google Search grounding -- regardless of the chat/image provider chosen below."
        )
        with gr.Row():
            chat_provider = gr.Dropdown(
                choices=list(CHAT_PROVIDERS), value=defaults.chat_provider, label="Chat provider"
            )
            chat_model_id = gr.Textbox(value=defaults.chat_model_id, label="Chat model ID")
        with gr.Row():
            image_provider = gr.Dropdown(
                choices=list(IMAGE_PROVIDERS), value=defaults.image_provider, label="Image provider"
            )
            image_model_id = gr.Textbox(value=defaults.image_model_id, label="Image model ID")
        gr.Markdown(
            "Some Gemini image models require the Gemini Enterprise Agent Platform "
            "(formerly Vertex AI) endpoint rather than the standard Gemini Developer "
            "API -- some API key restrictions only allow one or the other. If image "
            "generation fails with an access/permission error, enable this below; "
            "it still uses a plain API key (Express Mode), just a different endpoint. "
            "Only needed for the Gemini image provider."
        )
        with gr.Row():
            gemini_image_api_key = gr.Textbox(
                label="Gemini image API key (optional -- defaults to the Google API key "
                "below; set this if it needs a different key/restrictions than chat)",
                type="password",
            )
            gemini_image_enterprise = gr.Checkbox(
                value=defaults.gemini_image_enterprise,
                label="Use Gemini Enterprise/Vertex AI endpoint for image generation",
            )
        gr.Markdown(
            "Lets the Gemini image model search Google (including Google Images) "
            "for extra visual grounding at generation time, alongside Unsplash "
            "reference photos. Only applies to the Gemini image provider."
        )
        gemini_image_search_enabled = gr.Checkbox(
            value=defaults.gemini_image_search_enabled,
            label="Enable Google image search during generation",
        )

        google_api_key = gr.Textbox(label="Google API key (required)", type="password")
        with gr.Row():
            openai_api_key = gr.Textbox(label="OpenAI API key", type="password")
            anthropic_api_key = gr.Textbox(label="Anthropic API key", type="password")
            stability_api_key = gr.Textbox(label="Stability API key", type="password")
        unsplash_api_key = gr.Textbox(
            label="Unsplash Access Key (optional -- enables real reference photos, "
            "unsplash.com/developers)",
            type="password",
        )

        gr.Markdown(
            "Refinement mode picks how the generate/evaluate loop is driven. "
            "**fast** runs it as a plain loop and stops the moment an evaluation "
            "passes -- no extra model calls for orchestration. **thinking** drives "
            "it through an agent graph (an extra chat-model call per iteration): it "
            "writes its own generation prompt and can search Unsplash/the web "
            "mid-loop when it decides it needs more context. **instant** skips the "
            "loop and evaluation entirely, generating a single image."
        )
        with gr.Row():
            refinement_mode = gr.Dropdown(
                choices=list(REFINEMENT_MODES),
                value=defaults.refinement_mode,
                label="Refinement mode",
            )
            max_iterations = gr.Slider(
                1,
                MAX_ITERATIONS_CAP,
                value=defaults.max_iterations,
                step=1,
                label="Max iterations",
            )
            eval_pass_threshold = gr.Slider(
                1.0, 5.0, value=defaults.eval_pass_threshold, step=0.1, label="Pass threshold"
            )

        gr.Markdown(
            "SigLIP scoring adds an image/prompt similarity signal to evaluation, "
            "on top of Gemini's critique. Requires the optional dependencies "
            "(`uv sync --extra siglip`) -- if unavailable, it's silently skipped."
        )
        with gr.Row():
            siglip_enabled = gr.Checkbox(value=defaults.siglip_enabled, label="Enable SigLIP scoring")
            siglip_weight = gr.Slider(
                0.0,
                1.0,
                value=defaults.siglip_weight,
                step=0.05,
                label="SigLIP weight in overall score",
            )

    return [
        chat_provider,
        chat_model_id,
        image_provider,
        image_model_id,
        google_api_key,
        openai_api_key,
        anthropic_api_key,
        stability_api_key,
        unsplash_api_key,
        gemini_image_api_key,
        gemini_image_enterprise,
        gemini_image_search_enabled,
        refinement_mode,
        max_iterations,
        eval_pass_threshold,
        siglip_enabled,
        siglip_weight,
    ]


def fields_to_overlay(
    chat_provider: str,
    chat_model_id: str,
    image_provider: str,
    image_model_id: str,
    google_api_key: str,
    openai_api_key: str,
    anthropic_api_key: str,
    stability_api_key: str,
    unsplash_api_key: str,
    gemini_image_api_key: str,
    gemini_image_enterprise: bool,
    gemini_image_search_enabled: bool,
    refinement_mode: str,
    max_iterations: float | int,
    eval_pass_threshold: float,
    siglip_enabled: bool,
    siglip_weight: float,
) -> RuntimeSettingsOverlay:
    """Pure function: turns raw Gradio field values into a RuntimeSettingsOverlay.
    Blank strings become None (defer to env/default), matching resolve_settings'
    precedence contract. Testable without launching a Gradio server."""

    def blank_to_none(value: str) -> str | None:
        return value if value else None

    return RuntimeSettingsOverlay(
        chat_provider=blank_to_none(chat_provider),
        chat_model_id=blank_to_none(chat_model_id),
        image_provider=blank_to_none(image_provider),
        image_model_id=blank_to_none(image_model_id),
        google_api_key=blank_to_none(google_api_key),
        openai_api_key=blank_to_none(openai_api_key),
        anthropic_api_key=blank_to_none(anthropic_api_key),
        stability_api_key=blank_to_none(stability_api_key),
        unsplash_api_key=blank_to_none(unsplash_api_key),
        gemini_image_api_key=blank_to_none(gemini_image_api_key),
        gemini_image_enterprise=gemini_image_enterprise,
        gemini_image_search_enabled=gemini_image_search_enabled,
        refinement_mode=blank_to_none(refinement_mode),
        max_iterations=int(max_iterations) if max_iterations else None,
        eval_pass_threshold=float(eval_pass_threshold) if eval_pass_threshold else None,
        siglip_enabled=siglip_enabled,
        siglip_weight=float(siglip_weight) if siglip_weight else None,
    )
