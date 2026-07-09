from typing import Any

from google import genai
from google.genai import types

from pondercanvas.providers._gemini import gemini_http_options
from pondercanvas.providers._mime import sniff_image_mime
from pondercanvas.providers.image.base import ImageProvider, ImageResult


def _describe_missing_image(response: types.GenerateContentResponse) -> str:
    """Builds an actionable error message when Gemini returns no inline image
    data -- e.g. a safety block, a refusal, or an empty response -- instead of
    just the unhelpful generic message. Attribute access is defensive since
    exact response shape can vary across models/modes."""
    details: list[str] = []

    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
    if block_reason:
        details.append(f"prompt blocked: {block_reason}")

    candidates = response.candidates or []
    if not candidates:
        details.append("no candidates returned")
    else:
        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            details.append(f"finish_reason={finish_reason}")
        finish_message = getattr(candidate, "finish_message", None)
        if finish_message:
            details.append(f"finish_message={finish_message!r}")

        content = getattr(candidate, "content", None)
        text_parts = [
            part.text for part in (getattr(content, "parts", None) or []) if getattr(part, "text", None)
        ]
        if text_parts:
            details.append(f"model returned text instead of an image: {' '.join(text_parts)!r}")

    detail_str = "; ".join(details) if details else "no additional diagnostic info in response"
    return f"Gemini image generation returned no inline image data ({detail_str})"


class GeminiImageProvider(ImageProvider):
    """Sends prompt + optional reference images to a Gemini image model,
    extracts inline image bytes from the response."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None,
        aspect_ratio: str = "1:1",
        output_mime_type: str = "image/png",
        enterprise: bool = False,
    ):
        self.model_id = model_id
        self._api_key = api_key
        self.aspect_ratio = aspect_ratio
        self.output_mime_type = output_mime_type
        self._enterprise = enterprise
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            # enterprise=True routes through the Gemini Enterprise Agent
            # Platform (formerly Vertex AI) endpoint in "Express Mode" --
            # still authenticated with a plain API key, no service account/
            # ADC needed. Some models/API-key restrictions only permit one
            # of the two endpoints, so this must be switchable.
            self._client = genai.Client(
                api_key=self._api_key,
                enterprise=self._enterprise,
                http_options=gemini_http_options(),
            )
        return self._client

    def generate(self, prompt: str, reference_images: list[bytes], **params: Any) -> ImageResult:
        contents: list[Any] = [
            types.Part.from_bytes(data=image, mime_type=sniff_image_mime(image)) for image in reference_images
        ]
        contents.append(prompt)

        # response_modalities MUST be IMAGE-only, not ["TEXT", "IMAGE"]:
        # these models will happily return a prose *description* of the image
        # (finish_reason=STOP, no error) instead of drawing it if TEXT is an
        # allowed response modality -- especially on elaborate prompts with
        # reference images. Omitting TEXT removes that escape hatch and
        # reliably yields image bytes.
        #
        # output_mime_type is Enterprise/Vertex-only -- the Developer API
        # rejects it client-side (before any network call), so it's gated on
        # enterprise mode; aspect_ratio works on both.
        image_config_kwargs: dict[str, Any] = {
            "aspect_ratio": params.get("aspect_ratio", self.aspect_ratio),
        }
        if self._enterprise:
            image_config_kwargs["output_mime_type"] = self.output_mime_type

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**image_config_kwargs),
            ),
        )

        candidates = response.candidates or []
        content = candidates[0].content if candidates else None
        parts = content.parts if content is not None else None
        for part in parts or []:
            inline_data = part.inline_data
            if inline_data is not None and inline_data.data is not None:
                return ImageResult(
                    image_bytes=inline_data.data,
                    mime_type=inline_data.mime_type or self.output_mime_type,
                    provider="gemini",
                    model_id=self.model_id,
                )
        raise RuntimeError(_describe_missing_image(response))
