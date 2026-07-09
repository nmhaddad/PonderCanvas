from typing import Any

from pondercanvas.providers.image.base import ImageProvider, ImageResult


class FakeImageProvider(ImageProvider):
    """Scripted ImageProvider: returns results from `results` in order (or
    repeats the last one once exhausted), and records every call for
    assertions."""

    def __init__(self, results: list[ImageResult] | None = None):
        self.results = results or [
            ImageResult(
                image_bytes=b"fake-png",
                mime_type="image/png",
                provider="fake",
                model_id="m",
                interaction_id="fake-interaction-id",
            )
        ]
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, reference_images: list[bytes], **params: Any) -> ImageResult:
        self.calls.append(
            {"prompt": prompt, "reference_images": reference_images, "params": params}
        )
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[index]
