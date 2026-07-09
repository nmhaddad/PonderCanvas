import pytest

from pondercanvas.providers.image.gemini_image import GeminiImageProvider
from pondercanvas.providers.image.openai_image import OpenAIImageProvider
from pondercanvas.providers.image.registry import IMAGE_PROVIDER_REGISTRY, get_image_provider
from pondercanvas.providers.image.stability_image import StabilityImageProvider


class TestRegistryLookup:
    def test_gemini_returns_gemini_provider(self):
        provider = get_image_provider("gemini", model_id="m", api_key="k")
        assert isinstance(provider, GeminiImageProvider)

    def test_openai_returns_openai_provider(self):
        provider = get_image_provider("openai", model_id="m", api_key="k")
        assert isinstance(provider, OpenAIImageProvider)

    def test_stability_returns_stability_provider(self):
        provider = get_image_provider("stability", model_id="m", api_key="k")
        assert isinstance(provider, StabilityImageProvider)

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown image provider"):
            get_image_provider("does-not-exist", model_id="m", api_key="k")

    def test_registry_has_exactly_expected_providers(self):
        assert set(IMAGE_PROVIDER_REGISTRY) == {"gemini", "openai", "stability"}


class TestUnimplementedProviderStubs:
    def test_openai_generate_raises_not_implemented_with_informative_message(self):
        provider = OpenAIImageProvider(model_id="m", api_key="k")
        with pytest.raises(
            NotImplementedError, match="OpenAI image provider is not yet implemented"
        ):
            provider.generate("a prompt", [])

    def test_stability_generate_raises_not_implemented_with_informative_message(self):
        provider = StabilityImageProvider(model_id="m", api_key="k")
        with pytest.raises(
            NotImplementedError, match="Stability image provider is not yet implemented"
        ):
            provider.generate("a prompt", [])
