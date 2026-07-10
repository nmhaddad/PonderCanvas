from pondercanvas.config.settings import AppSettings, RuntimeSettingsOverlay, resolve_settings
from pondercanvas.ui.settings_panel import fields_to_overlay


def _call(**overrides):
    defaults = dict(
        chat_model_id="",
        image_model_id="",
        unsplash_api_key="",
        gemini_image_api_key="",
        gemini_image_enterprise=False,
        gemini_image_search_enabled=True,
        refinement_mode="",
        max_iterations=0,
        eval_pass_threshold=0.0,
        siglip_enabled=False,
        siglip_weight=0.0,
    )
    defaults.update(overrides)
    return fields_to_overlay(**defaults)


class TestFieldsToOverlay:
    def test_all_blank_produces_empty_overlay(self):
        # siglip_enabled/gemini_image_enterprise/gemini_image_search_enabled
        # are checkboxes, not nullable text/number fields: they always pass
        # their current value through.
        overlay = _call()
        assert overlay == RuntimeSettingsOverlay(
            siglip_enabled=False,
            gemini_image_enterprise=False,
            gemini_image_search_enabled=True,
        )

    def test_blank_string_becomes_none(self):
        overlay = _call(chat_model_id="")
        assert overlay.chat_model_id is None

    def test_filled_string_passes_through(self):
        overlay = _call(chat_model_id="gemini-custom")
        assert overlay.chat_model_id == "gemini-custom"

    def test_zero_max_iterations_becomes_none_not_zero(self):
        # 0 is falsy: Gradio's slider default/unset should defer to env/default,
        # not force max_iterations to 0.
        overlay = _call(max_iterations=0)
        assert overlay.max_iterations is None

    def test_nonzero_max_iterations_becomes_int(self):
        overlay = _call(max_iterations=3.0)
        assert overlay.max_iterations == 3
        assert isinstance(overlay.max_iterations, int)

    def test_eval_pass_threshold_becomes_float(self):
        overlay = _call(eval_pass_threshold=4.5)
        assert overlay.eval_pass_threshold == 4.5

    def test_returns_type_runtime_settings_overlay(self):
        assert isinstance(_call(), RuntimeSettingsOverlay)

    def test_siglip_enabled_true_passes_through(self):
        overlay = _call(siglip_enabled=True)
        assert overlay.siglip_enabled is True

    def test_siglip_enabled_false_passes_through(self):
        overlay = _call(siglip_enabled=False)
        assert overlay.siglip_enabled is False

    def test_zero_siglip_weight_becomes_none_not_zero(self):
        overlay = _call(siglip_weight=0.0)
        assert overlay.siglip_weight is None

    def test_nonzero_siglip_weight_becomes_float(self):
        overlay = _call(siglip_weight=0.4)
        assert overlay.siglip_weight == 0.4

    def test_unsplash_api_key_passes_through(self):
        overlay = _call(unsplash_api_key="typed-in-ui")
        assert overlay.unsplash_api_key == "typed-in-ui"

    def test_blank_unsplash_api_key_becomes_none(self):
        overlay = _call(unsplash_api_key="")
        assert overlay.unsplash_api_key is None

    def test_gemini_image_api_key_passes_through(self):
        overlay = _call(gemini_image_api_key="typed-in-ui")
        assert overlay.gemini_image_api_key == "typed-in-ui"

    def test_blank_gemini_image_api_key_becomes_none(self):
        overlay = _call(gemini_image_api_key="")
        assert overlay.gemini_image_api_key is None

    def test_gemini_image_enterprise_true_passes_through(self):
        overlay = _call(gemini_image_enterprise=True)
        assert overlay.gemini_image_enterprise is True

    def test_gemini_image_enterprise_false_passes_through(self):
        overlay = _call(gemini_image_enterprise=False)
        assert overlay.gemini_image_enterprise is False

    def test_gemini_image_search_enabled_true_passes_through(self):
        overlay = _call(gemini_image_search_enabled=True)
        assert overlay.gemini_image_search_enabled is True

    def test_gemini_image_search_enabled_false_passes_through(self):
        overlay = _call(gemini_image_search_enabled=False)
        assert overlay.gemini_image_search_enabled is False

    def test_refinement_mode_passes_through(self):
        overlay = _call(refinement_mode="thinking")
        assert overlay.refinement_mode == "thinking"

    def test_blank_refinement_mode_becomes_none(self):
        overlay = _call(refinement_mode="")
        assert overlay.refinement_mode is None


class TestOverlayFeedsIntoResolveSettings:
    def test_overlay_from_ui_fields_overrides_env_defaults(self):
        overlay = _call(chat_model_id="gemini-custom", gemini_image_api_key="ui-key")
        base = AppSettings(_env_file=None, chat_model_id="gemini-3.5-flash")  # type: ignore[call-arg]
        effective = resolve_settings(base, overlay)
        assert effective.chat_model_id == "gemini-custom"
        assert effective.gemini_image_api_key == "ui-key"

    def test_all_blank_ui_fields_defer_entirely_to_env(self):
        # chat_provider has no UI field anymore -- it can only come from env,
        # confirming the overlay doesn't clobber it with a blank/default.
        overlay = _call()
        base = AppSettings(_env_file=None, chat_provider="openai")  # type: ignore[call-arg]
        effective = resolve_settings(base, overlay)
        assert effective.chat_provider == "openai"
