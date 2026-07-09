from abc import ABC, abstractmethod

from pydantic import BaseModel


class StructuredVisionProvider(ABC):
    @abstractmethod
    def generate_structured[T: BaseModel](
        self, prompt: str, images: list[bytes], response_schema: type[T]
    ) -> T: ...
