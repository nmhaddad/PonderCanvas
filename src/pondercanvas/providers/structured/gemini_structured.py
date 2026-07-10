from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from pondercanvas.providers._gemini import gemini_http_options
from pondercanvas.providers._mime import sniff_image_mime
from pondercanvas.providers.structured.base import StructuredVisionProvider


class GeminiStructuredVisionProvider(StructuredVisionProvider):
    """Sends prompt + images to Gemini with a Pydantic response_schema, parses
    the resulting structured JSON. Used for both brief extraction and image
    evaluation, independent of the user's chosen chat/image provider."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        # Always Enterprise/Vertex AI + Application Default Credentials --
        # see AppSettings' comment near gemini_image_api_key for why this
        # doesn't cover image generation.
        if self._client is None:
            self._client = genai.Client(enterprise=True, http_options=gemini_http_options())
        return self._client

    def generate_structured[T: BaseModel](
        self, prompt: str, images: list[bytes], response_schema: type[T]
    ) -> T:
        contents: list[Any] = [
            types.Part.from_bytes(data=image, mime_type=sniff_image_mime(image)) for image in images
        ]
        contents.append(prompt)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        if response.text is None:
            raise RuntimeError("Gemini structured generation returned no text content")
        return response_schema.model_validate_json(response.text)
