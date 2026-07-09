from typing import Any

from pondercanvas.providers.image.base import ImageProvider, ImageResult


class StabilityImageProvider(ImageProvider):
    """Not yet implemented: no Stability API key/testing available at design
    time. Wired into the registry so selecting it fails loudly and clearly
    instead of silently falling back to another provider. Implement by
    following the ImageProvider contract in base.py."""

    def __init__(self, model_id: str, api_key: str | None, **kwargs: Any):
        self.model_id = model_id
        self._api_key = api_key

    def generate(self, prompt: str, reference_images: list[bytes], **params: Any) -> ImageResult:
        raise NotImplementedError(
            "Stability image provider is not yet implemented. "
            "Implement ImageProvider.generate in providers/image/stability_image.py."
        )
