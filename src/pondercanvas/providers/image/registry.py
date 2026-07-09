from typing import Any

from pondercanvas.providers.image.base import ImageProvider
from pondercanvas.providers.image.gemini_image import GeminiImageProvider
from pondercanvas.providers.image.openai_image import OpenAIImageProvider
from pondercanvas.providers.image.stability_image import StabilityImageProvider

IMAGE_PROVIDER_REGISTRY: dict[str, type[ImageProvider]] = {
    "gemini": GeminiImageProvider,
    "openai": OpenAIImageProvider,
    "stability": StabilityImageProvider,
}


def get_image_provider(name: str, **kwargs: Any) -> ImageProvider:
    try:
        provider_cls = IMAGE_PROVIDER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(IMAGE_PROVIDER_REGISTRY))
        raise ValueError(f"Unknown image provider {name!r}; available: {available}") from None
    return provider_cls(**kwargs)
