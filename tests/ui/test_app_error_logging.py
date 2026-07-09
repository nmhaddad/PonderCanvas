import logging

import pytest

from pondercanvas.config.settings import AppSettings
from pondercanvas.ui import app as app_module

# Positional order must match settings_panel.SETTINGS_FIELD_ORDER.
_SETTINGS_FIELD_VALUES = (
    "",  # chat_provider
    "",  # chat_model_id
    "",  # image_provider
    "",  # image_model_id
    "test-google-key",  # google_api_key
    "",  # openai_api_key
    "",  # anthropic_api_key
    "",  # stability_api_key
    "",  # unsplash_api_key
    "",  # gemini_image_api_key
    False,  # gemini_image_enterprise
    0,  # max_iterations
    0.0,  # eval_pass_threshold
    False,  # siglip_enabled
    0.0,  # siglip_weight
)


class _BoomPipeline:
    def __init__(self, settings):
        pass

    async def run(self, prompt, reference_images):
        raise RuntimeError("boom")


class TestOnGenerateErrorLogging:
    async def test_pipeline_failure_is_logged_with_traceback_and_reraised(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(app_module, "PonderCanvasPipeline", _BoomPipeline)

        with caplog.at_level(logging.ERROR, logger="pondercanvas.ui.app"):
            with pytest.raises(RuntimeError, match="boom"):
                await app_module._on_generate("draw a cat", None, *_SETTINGS_FIELD_VALUES)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "draw a cat" in record.getMessage()
        assert record.exc_info is not None  # logger.exception() captured the traceback

    async def test_missing_google_key_does_not_log_anything(self, monkeypatch, caplog):
        monkeypatch.setattr(app_module, "PonderCanvasPipeline", _BoomPipeline)
        # Isolate from the developer's real .env (which may legitimately have
        # a real key configured for actually running the app locally).
        monkeypatch.setattr(app_module, "AppSettings", lambda: AppSettings(_env_file=None))
        blank_key_values = (*_SETTINGS_FIELD_VALUES[:4], "", *_SETTINGS_FIELD_VALUES[5:])

        with caplog.at_level(logging.ERROR, logger="pondercanvas.ui.app"):
            result = await app_module._on_generate("draw a cat", None, *blank_key_values)

        assert result == (None, app_module._MISSING_GOOGLE_KEY_MESSAGE)
        assert caplog.records == []
