from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm

from pondercanvas.agent.pipeline import PonderCanvasPipeline
from pondercanvas.config.settings import AppSettings, resolve_settings
from pondercanvas.providers.chat.factory import build_chat_model
from pondercanvas.providers.image.gemini_image import GeminiImageProvider
from pondercanvas.providers.image.openai_image import OpenAIImageProvider
from pondercanvas.providers.structured.gemini_structured import GeminiStructuredVisionProvider


def _effective(tmp_path, **overrides):
    defaults = dict(output_dir=tmp_path, google_api_key="fake-key")
    defaults.update(overrides)
    return resolve_settings(AppSettings(_env_file=None, **defaults))  # type: ignore[call-arg]


class TestPipelineProviderSelection:
    def test_default_image_provider_is_gemini(self, tmp_path):
        pipeline = PonderCanvasPipeline(_effective(tmp_path))
        assert isinstance(pipeline.image_provider, GeminiImageProvider)

    def test_switching_image_provider_changes_concrete_class(self, tmp_path):
        pipeline = PonderCanvasPipeline(
            _effective(tmp_path, image_provider="openai", openai_api_key="o-key")
        )
        assert isinstance(pipeline.image_provider, OpenAIImageProvider)

    def test_structured_provider_is_always_gemini_regardless_of_chat_provider(self, tmp_path):
        pipeline = PonderCanvasPipeline(_effective(tmp_path, chat_provider="openai"))
        assert isinstance(pipeline.structured_provider, GeminiStructuredVisionProvider)

    def test_image_provider_receives_correct_api_key_per_provider(self, tmp_path):
        pipeline = PonderCanvasPipeline(
            _effective(
                tmp_path,
                image_provider="gemini",
                google_api_key="g-key-for-image",
            )
        )
        assert pipeline.image_provider._api_key == "g-key-for-image"

        pipeline2 = PonderCanvasPipeline(
            _effective(tmp_path, image_provider="openai", openai_api_key="o-key-for-image")
        )
        assert pipeline2.image_provider._api_key == "o-key-for-image"

    def test_gemini_image_provider_uses_distinct_key_when_configured(self, tmp_path):
        pipeline = PonderCanvasPipeline(
            _effective(
                tmp_path,
                google_api_key="shared-key",
                gemini_image_api_key="image-only-key",
            )
        )
        assert pipeline.image_provider._api_key == "image-only-key"

    def test_gemini_image_enterprise_flag_propagates_to_provider(self, tmp_path):
        pipeline = PonderCanvasPipeline(_effective(tmp_path, gemini_image_enterprise=True))
        assert pipeline.image_provider._enterprise is True

        pipeline2 = PonderCanvasPipeline(_effective(tmp_path, gemini_image_enterprise=False))
        assert pipeline2.image_provider._enterprise is False

    def test_build_chat_model_gemini(self, tmp_path):
        settings = _effective(tmp_path, chat_provider="gemini")
        assert isinstance(build_chat_model(settings), Gemini)

    def test_build_chat_model_openai_uses_litellm(self, tmp_path):
        settings = _effective(tmp_path, chat_provider="openai", openai_api_key="o-key")
        assert isinstance(build_chat_model(settings), LiteLlm)

    def test_no_network_touched_by_provider_construction(self, tmp_path):
        # Constructing providers/pipeline must never make a live network
        # call; only .generate()/.run() would. This just exercises
        # construction paths for every provider combination without error.
        for image_provider, chat_provider in [
            ("gemini", "gemini"),
            ("openai", "openai"),
            ("stability", "anthropic"),
        ]:
            settings = _effective(
                tmp_path,
                image_provider=image_provider,
                chat_provider=chat_provider,
                openai_api_key="o-key",
                anthropic_api_key="a-key",
                stability_api_key="s-key",
            )
            pipeline = PonderCanvasPipeline(settings)
            assert pipeline.image_provider is not None
