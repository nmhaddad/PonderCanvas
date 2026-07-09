from typing import Any

from pydantic import BaseModel

from pondercanvas.providers.structured.base import StructuredVisionProvider


class FakeStructuredVisionProvider(StructuredVisionProvider):
    """Scripted StructuredVisionProvider: returns results from `results` in
    order (or repeats the last one once exhausted), and records every call
    for assertions."""

    def __init__(self, results: list[BaseModel]):
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def generate_structured[T: BaseModel](
        self, prompt: str, images: list[bytes], response_schema: type[T]
    ) -> T:
        self.calls.append({"prompt": prompt, "images": images, "response_schema": response_schema})
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[index]  # type: ignore[return-value]
