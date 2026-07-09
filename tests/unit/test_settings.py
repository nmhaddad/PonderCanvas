from pathlib import Path

from pondercanvas.config.constants import MAX_ITERATIONS_CAP
from pondercanvas.config.settings import (
    AppSettings,
    RuntimeSettingsOverlay,
    resolve_settings,
)


def _base(**overrides) -> AppSettings:
    # _env_file=None: isolate from any real .env / OS env vars so tests are
    # deterministic regardless of the developer's local environment.
    return AppSettings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestResolveSettingsPrecedence:
    def test_defaults_used_when_no_overlay(self):
        effective = resolve_settings(_base())
        assert effective.chat_provider == "gemini"
        assert effective.max_iterations == 3

    def test_env_derived_base_overrides_default(self):
        base = _base(chat_provider="openai", chat_model_id="gpt-5")
        effective = resolve_settings(base)
        assert effective.chat_provider == "openai"
        assert effective.chat_model_id == "gpt-5"

    def test_overlay_wins_over_base(self):
        base = _base(chat_provider="openai")
        overlay = RuntimeSettingsOverlay(chat_provider="anthropic")
        effective = resolve_settings(base, overlay)
        assert effective.chat_provider == "anthropic"

    def test_overlay_none_fields_defer_to_base(self):
        base = _base(image_model_id="from-env")
        overlay = RuntimeSettingsOverlay(chat_provider="anthropic")  # image_model_id untouched
        effective = resolve_settings(base, overlay)
        assert effective.image_model_id == "from-env"
        assert effective.chat_provider == "anthropic"

    def test_no_overlay_argument_behaves_like_empty_overlay(self):
        base = _base(chat_provider="openai")
        assert resolve_settings(base).chat_provider == "openai"
        assert resolve_settings(base, None).chat_provider == "openai"


class TestMaxIterationsCap:
    def test_base_value_above_cap_is_clamped(self):
        base = _base(max_iterations=99)
        effective = resolve_settings(base)
        assert effective.max_iterations == MAX_ITERATIONS_CAP

    def test_overlay_value_above_cap_is_clamped(self):
        base = _base()
        overlay = RuntimeSettingsOverlay(max_iterations=100)
        effective = resolve_settings(base, overlay)
        assert effective.max_iterations == MAX_ITERATIONS_CAP

    def test_value_below_one_is_clamped_to_one(self):
        base = _base(max_iterations=0)
        effective = resolve_settings(base)
        assert effective.max_iterations == 1

    def test_default_is_three(self):
        effective = resolve_settings(_base())
        assert effective.max_iterations == 3


class TestApiKeySecretHandling:
    def test_base_secret_str_is_unwrapped_to_plain_str(self):
        base = _base(google_api_key="secret-from-env")
        effective = resolve_settings(base)
        assert effective.google_api_key == "secret-from-env"
        assert isinstance(effective.google_api_key, str)

    def test_overlay_key_overrides_base_key(self):
        base = _base(google_api_key="env-key")
        overlay = RuntimeSettingsOverlay(google_api_key="ui-typed-key")
        effective = resolve_settings(base, overlay)
        assert effective.google_api_key == "ui-typed-key"

    def test_missing_key_is_none(self):
        effective = resolve_settings(_base())
        assert effective.openai_api_key is None

    def test_repr_of_app_settings_does_not_leak_secret(self):
        base = _base(google_api_key="super-secret-value")
        assert "super-secret-value" not in repr(base)
        assert "super-secret-value" not in str(base.google_api_key)

    def test_unsplash_key_overlay_overrides_base(self):
        base = _base(unsplash_api_key="env-key")
        overlay = RuntimeSettingsOverlay(unsplash_api_key="ui-key")
        effective = resolve_settings(base, overlay)
        assert effective.unsplash_api_key == "ui-key"

    def test_missing_unsplash_key_is_none(self):
        effective = resolve_settings(_base())
        assert effective.unsplash_api_key is None


class TestGeminiImageApiKeyFallback:
    def test_falls_back_to_google_api_key_when_not_configured(self):
        base = _base(google_api_key="shared-key")
        effective = resolve_settings(base)
        assert effective.gemini_image_api_key == "shared-key"

    def test_distinct_env_key_is_used_over_the_shared_one(self):
        base = _base(google_api_key="shared-key", gemini_image_api_key="image-only-key")
        effective = resolve_settings(base)
        assert effective.gemini_image_api_key == "image-only-key"
        assert effective.google_api_key == "shared-key"

    def test_overlay_distinct_key_wins_over_base(self):
        base = _base(google_api_key="shared-key", gemini_image_api_key="base-image-key")
        overlay = RuntimeSettingsOverlay(gemini_image_api_key="ui-image-key")
        effective = resolve_settings(base, overlay)
        assert effective.gemini_image_api_key == "ui-image-key"

    def test_both_unset_is_none(self):
        effective = resolve_settings(_base())
        assert effective.gemini_image_api_key is None


class TestGeminiImageEnterpriseSetting:
    def test_defaults_to_disabled(self):
        effective = resolve_settings(_base())
        assert effective.gemini_image_enterprise is False

    def test_env_derived_base_can_enable_it(self):
        base = _base(gemini_image_enterprise=True)
        effective = resolve_settings(base)
        assert effective.gemini_image_enterprise is True

    def test_overlay_can_enable_it_over_a_disabled_base(self):
        base = _base(gemini_image_enterprise=False)
        overlay = RuntimeSettingsOverlay(gemini_image_enterprise=True)
        effective = resolve_settings(base, overlay)
        assert effective.gemini_image_enterprise is True


class TestDownloadLimitSettings:
    def test_defaults_are_used(self):
        effective = resolve_settings(_base())
        assert effective.max_reference_downloads == 3
        assert effective.max_download_bytes == 5_000_000
        assert effective.download_timeout_s == 5.0

    def test_env_derived_values_are_used(self):
        base = _base(max_reference_downloads=1, max_download_bytes=999, download_timeout_s=2.5)
        effective = resolve_settings(base)
        assert effective.max_reference_downloads == 1
        assert effective.max_download_bytes == 999
        assert effective.download_timeout_s == 2.5


class TestRedaction:
    def test_redacted_masks_all_api_key_fields(self):
        base = _base(
            google_api_key="g-key", openai_api_key="o-key", anthropic_api_key="a-key"
        )
        effective = resolve_settings(base)
        redacted = effective.redacted()
        assert redacted["google_api_key"] == "***REDACTED***"
        assert redacted["openai_api_key"] == "***REDACTED***"
        assert redacted["anthropic_api_key"] == "***REDACTED***"
        assert redacted["stability_api_key"] is None

    def test_redacted_masks_unsplash_api_key(self):
        base = _base(unsplash_api_key="u-key")
        effective = resolve_settings(base)
        redacted = effective.redacted()
        assert redacted["unsplash_api_key"] == "***REDACTED***"

    def test_redacted_masks_gemini_image_api_key_fallback_too(self):
        base = _base(google_api_key="g-key")
        effective = resolve_settings(base)
        redacted = effective.redacted()
        assert redacted["gemini_image_api_key"] == "***REDACTED***"

    def test_redacted_preserves_non_secret_fields(self):
        base = _base(chat_provider="anthropic", max_iterations=3)
        effective = resolve_settings(base)
        redacted = effective.redacted()
        assert redacted["chat_provider"] == "anthropic"
        assert redacted["max_iterations"] == 3

    def test_redacted_does_not_mutate_original(self):
        base = _base(google_api_key="g-key")
        effective = resolve_settings(base)
        effective.redacted()
        assert effective.google_api_key == "g-key"


class TestRefinementMode:
    def test_defaults_to_fast(self):
        effective = resolve_settings(_base())
        assert effective.refinement_mode == "fast"

    def test_env_derived_base_can_select_thinking(self):
        base = _base(refinement_mode="thinking")
        effective = resolve_settings(base)
        assert effective.refinement_mode == "thinking"

    def test_overlay_wins_over_base(self):
        base = _base(refinement_mode="fast")
        overlay = RuntimeSettingsOverlay(refinement_mode="thinking")
        effective = resolve_settings(base, overlay)
        assert effective.refinement_mode == "thinking"

    def test_unknown_mode_falls_back_to_default(self):
        base = _base(refinement_mode="nonsense")
        effective = resolve_settings(base)
        assert effective.refinement_mode == "fast"

    def test_overlay_none_defers_to_base(self):
        base = _base(refinement_mode="thinking")
        overlay = RuntimeSettingsOverlay(chat_provider="anthropic")  # refinement_mode untouched
        effective = resolve_settings(base, overlay)
        assert effective.refinement_mode == "thinking"


class TestNonOverridableFields:
    def test_structured_model_id_always_comes_from_base(self):
        # No overlay field exists for structured_model_id: extraction/eval
        # always uses Gemini per the architecture decision, never swappable.
        assert "structured_model_id" not in RuntimeSettingsOverlay.model_fields
        base = _base(structured_model_id="gemini-custom")
        effective = resolve_settings(base)
        assert effective.structured_model_id == "gemini-custom"


def test_effective_settings_output_dir_is_path():
    effective = resolve_settings(_base())
    assert isinstance(effective.output_dir, Path)


class TestSiglipSettings:
    def test_defaults_to_disabled(self):
        effective = resolve_settings(_base())
        assert effective.siglip_enabled is False

    def test_env_derived_base_can_enable_it(self):
        base = _base(siglip_enabled=True)
        effective = resolve_settings(base)
        assert effective.siglip_enabled is True

    def test_overlay_can_enable_it_over_a_disabled_base(self):
        base = _base(siglip_enabled=False)
        overlay = RuntimeSettingsOverlay(siglip_enabled=True)
        effective = resolve_settings(base, overlay)
        assert effective.siglip_enabled is True

    def test_overlay_none_defers_to_base_enabled_value(self):
        base = _base(siglip_enabled=True)
        overlay = RuntimeSettingsOverlay(chat_provider="anthropic")  # siglip_enabled untouched
        effective = resolve_settings(base, overlay)
        assert effective.siglip_enabled is True

    def test_weight_above_one_is_clamped_to_one(self):
        base = _base(siglip_weight=2.0)
        effective = resolve_settings(base)
        assert effective.siglip_weight == 1.0

    def test_weight_below_zero_is_clamped_to_zero(self):
        base = _base(siglip_weight=-0.5)
        effective = resolve_settings(base)
        assert effective.siglip_weight == 0.0

    def test_overlay_weight_is_clamped_too(self):
        base = _base()
        overlay = RuntimeSettingsOverlay(siglip_weight=5.0)
        effective = resolve_settings(base, overlay)
        assert effective.siglip_weight == 1.0
