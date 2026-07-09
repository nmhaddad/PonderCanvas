import gradio as gr

from pondercanvas.ui.settings_panel import SETTINGS_FIELD_ORDER, build_settings_panel


def _build_fields_by_name() -> dict:
    with gr.Blocks():
        fields = build_settings_panel()
    return dict(zip(SETTINGS_FIELD_ORDER, fields, strict=True))


class TestBuildSettingsPanelReflectsEnvDefaults:
    """Regression test: checkboxes (and other non-secret fields) must be
    seeded from the actual resolved env/.env settings, not a hardcoded
    constant -- otherwise an untouched checkbox always submits its default
    value and permanently overrides whatever was configured via env/.env
    (see RuntimeSettingsOverlay/resolve_settings precedence)."""

    def test_gemini_image_enterprise_checkbox_reflects_env_true(self, monkeypatch):
        monkeypatch.setenv("PONDERCANVAS_GEMINI_IMAGE_ENTERPRISE", "true")
        fields = _build_fields_by_name()
        assert fields["gemini_image_enterprise"].value is True

    def test_gemini_image_enterprise_checkbox_reflects_env_false(self, monkeypatch):
        monkeypatch.setenv("PONDERCANVAS_GEMINI_IMAGE_ENTERPRISE", "false")
        fields = _build_fields_by_name()
        assert fields["gemini_image_enterprise"].value is False

    def test_siglip_enabled_checkbox_reflects_env_true(self, monkeypatch):
        monkeypatch.setenv("PONDERCANVAS_SIGLIP_ENABLED", "true")
        fields = _build_fields_by_name()
        assert fields["siglip_enabled"].value is True

    def test_chat_provider_dropdown_reflects_env_value(self, monkeypatch):
        monkeypatch.setenv("PONDERCANVAS_CHAT_PROVIDER", "anthropic")
        fields = _build_fields_by_name()
        assert fields["chat_provider"].value == "anthropic"

    def test_max_iterations_slider_reflects_env_value(self, monkeypatch):
        monkeypatch.setenv("PONDERCANVAS_MAX_ITERATIONS", "2")
        fields = _build_fields_by_name()
        assert fields["max_iterations"].value == 2

    def test_api_key_fields_stay_blank_even_when_set_in_env(self, monkeypatch):
        # Secrets must never be pre-filled into page HTML/DOM.
        monkeypatch.setenv("PONDERCANVAS_GOOGLE_API_KEY", "should-not-appear-in-page")
        fields = _build_fields_by_name()
        assert fields["google_api_key"].value != "should-not-appear-in-page"
