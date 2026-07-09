from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from pondercanvas.config.constants import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_CHAT_MODEL_ID,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_DOWNLOAD_TIMEOUT_S,
    DEFAULT_EVAL_PASS_THRESHOLD,
    DEFAULT_GEMINI_IMAGE_ENTERPRISE,
    DEFAULT_IMAGE_MODEL_ID,
    DEFAULT_IMAGE_PROVIDER,
    DEFAULT_MAX_DOWNLOAD_BYTES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_REFERENCE_DOWNLOADS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFINEMENT_MODE,
    DEFAULT_SIGLIP_ENABLED,
    DEFAULT_SIGLIP_WEIGHT,
    DEFAULT_STRUCTURED_MODEL_ID,
    MAX_ITERATIONS_CAP,
    REDACTED_MARKER,
    REFINEMENT_MODES,
    SECRET_FIELD_SUFFIX,
)


class AppSettings(BaseSettings):
    """Process-level defaults, read once from env vars / .env at startup."""

    model_config = SettingsConfigDict(
        env_prefix="PONDERCANVAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chat_provider: str = DEFAULT_CHAT_PROVIDER
    chat_model_id: str = DEFAULT_CHAT_MODEL_ID
    image_provider: str = DEFAULT_IMAGE_PROVIDER
    image_model_id: str = DEFAULT_IMAGE_MODEL_ID
    structured_model_id: str = DEFAULT_STRUCTURED_MODEL_ID

    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    stability_api_key: SecretStr | None = None
    unsplash_api_key: SecretStr | None = None
    gemini_image_api_key: SecretStr | None = None
    gemini_image_enterprise: bool = DEFAULT_GEMINI_IMAGE_ENTERPRISE

    refinement_mode: str = DEFAULT_REFINEMENT_MODE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    eval_pass_threshold: float = DEFAULT_EVAL_PASS_THRESHOLD
    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    max_reference_downloads: int = DEFAULT_MAX_REFERENCE_DOWNLOADS
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES
    download_timeout_s: float = DEFAULT_DOWNLOAD_TIMEOUT_S
    output_dir: Path = DEFAULT_OUTPUT_DIR

    siglip_enabled: bool = DEFAULT_SIGLIP_ENABLED
    siglip_weight: float = DEFAULT_SIGLIP_WEIGHT


class RuntimeSettingsOverlay(BaseModel):
    """Per-browser-session edits made live in the Gradio Settings panel.

    Every field is Optional: None means "not overridden, defer to AppSettings"."""

    chat_provider: str | None = None
    chat_model_id: str | None = None
    image_provider: str | None = None
    image_model_id: str | None = None

    google_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    stability_api_key: str | None = None
    unsplash_api_key: str | None = None
    gemini_image_api_key: str | None = None
    gemini_image_enterprise: bool | None = None

    refinement_mode: str | None = None
    max_iterations: int | None = None
    eval_pass_threshold: float | None = None

    siglip_enabled: bool | None = None
    siglip_weight: float | None = None


class EffectiveSettings(BaseModel):
    """Fully resolved, immutable config consumed by the rest of the app."""

    chat_provider: str
    chat_model_id: str
    image_provider: str
    image_model_id: str
    structured_model_id: str

    google_api_key: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None
    stability_api_key: str | None
    unsplash_api_key: str | None
    gemini_image_api_key: str | None
    gemini_image_enterprise: bool

    refinement_mode: str
    max_iterations: int
    eval_pass_threshold: float
    aspect_ratio: str
    max_reference_downloads: int
    max_download_bytes: int
    download_timeout_s: float
    output_dir: Path

    siglip_enabled: bool
    siglip_weight: float

    def redacted(self) -> dict:
        """Dict safe for logging/trace persistence: *_api_key fields masked."""
        data = self.model_dump(mode="json")
        for key, value in data.items():
            if key.endswith(SECRET_FIELD_SUFFIX) and value is not None:
                data[key] = REDACTED_MARKER
        return data


def _pick[T](overlay_value: T | None, base_value: T) -> T:
    return overlay_value if overlay_value is not None else base_value


def _secret_value(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret is not None else None


def resolve_settings(
    base: AppSettings, overlay: RuntimeSettingsOverlay | None = None
) -> EffectiveSettings:
    """Precedence: overlay > env-derived base > default. Pure function, no I/O."""
    overlay = overlay or RuntimeSettingsOverlay()

    max_iterations = _pick(overlay.max_iterations, base.max_iterations)
    max_iterations = max(1, min(max_iterations, MAX_ITERATIONS_CAP))

    siglip_weight = _pick(overlay.siglip_weight, base.siglip_weight)
    siglip_weight = max(0.0, min(siglip_weight, 1.0))

    refinement_mode = _pick(overlay.refinement_mode, base.refinement_mode)
    if refinement_mode not in REFINEMENT_MODES:
        refinement_mode = DEFAULT_REFINEMENT_MODE

    google_api_key = _pick(overlay.google_api_key, _secret_value(base.google_api_key))
    # Gemini image generation sometimes needs a distinct key from the shared
    # one (e.g. Enterprise/Vertex mode vs. Generative Language API key
    # restrictions aren't always combinable on one key): fall back to the
    # shared key only when no distinct one is configured.
    gemini_image_api_key = (
        _pick(overlay.gemini_image_api_key, _secret_value(base.gemini_image_api_key))
        or google_api_key
    )

    return EffectiveSettings(
        chat_provider=_pick(overlay.chat_provider, base.chat_provider),
        chat_model_id=_pick(overlay.chat_model_id, base.chat_model_id),
        image_provider=_pick(overlay.image_provider, base.image_provider),
        image_model_id=_pick(overlay.image_model_id, base.image_model_id),
        structured_model_id=base.structured_model_id,
        google_api_key=google_api_key,
        openai_api_key=_pick(overlay.openai_api_key, _secret_value(base.openai_api_key)),
        anthropic_api_key=_pick(overlay.anthropic_api_key, _secret_value(base.anthropic_api_key)),
        stability_api_key=_pick(overlay.stability_api_key, _secret_value(base.stability_api_key)),
        unsplash_api_key=_pick(overlay.unsplash_api_key, _secret_value(base.unsplash_api_key)),
        gemini_image_api_key=gemini_image_api_key,
        gemini_image_enterprise=_pick(
            overlay.gemini_image_enterprise, base.gemini_image_enterprise
        ),
        refinement_mode=refinement_mode,
        max_iterations=max_iterations,
        eval_pass_threshold=_pick(overlay.eval_pass_threshold, base.eval_pass_threshold),
        aspect_ratio=base.aspect_ratio,
        max_reference_downloads=base.max_reference_downloads,
        max_download_bytes=base.max_download_bytes,
        download_timeout_s=base.download_timeout_s,
        output_dir=base.output_dir,
        siglip_enabled=_pick(overlay.siglip_enabled, base.siglip_enabled),
        siglip_weight=siglip_weight,
    )
