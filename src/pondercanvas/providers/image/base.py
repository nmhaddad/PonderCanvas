from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class ImageResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_bytes: bytes
    mime_type: str
    provider: str
    model_id: str
    metadata: dict[str, Any] = {}


class ImageProvider(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, reference_images: list[bytes], **params: Any
    ) -> ImageResult: ...
