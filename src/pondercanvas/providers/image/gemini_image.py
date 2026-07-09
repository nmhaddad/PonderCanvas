import base64
from typing import Any

from google import genai
from google.genai import interactions

from pondercanvas.providers._gemini import gemini_http_options
from pondercanvas.providers._mime import sniff_image_mime
from pondercanvas.providers.image.base import ImageProvider, ImageResult


def _describe_missing_image(interaction: interactions.Interaction) -> str:
    """Builds an actionable error message when Gemini returns no output image
    -- e.g. a safety block, a refusal, or an empty response -- instead of just
    the unhelpful generic message. Attribute access is defensive since exact
    response shape can vary across models/modes."""
    details: list[str] = []

    status = getattr(interaction, "status", None)
    if status and status != "completed":
        details.append(f"status={status}")

    steps = getattr(interaction, "steps", None) or []
    text_parts: list[str] = []
    for step in steps:
        for block in getattr(step, "content", None) or []:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)
    if text_parts:
        details.append(f"model returned text instead of an image: {' '.join(text_parts)!r}")

    output_text = getattr(interaction, "output_text", None)
    if output_text and output_text not in text_parts:
        details.append(f"model returned text instead of an image: {output_text!r}")

    detail_str = "; ".join(details) if details else "no additional diagnostic info in response"
    return f"Gemini image generation returned no output image ({detail_str})"


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
        image_search_enabled: bool = True,
    ):
        self.model_id = model_id
        self._api_key = api_key
        self.aspect_ratio = aspect_ratio
        self.output_mime_type = output_mime_type
        self._enterprise = enterprise
        self._image_search_enabled = image_search_enabled
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
        input_content: list[dict[str, Any]] = [
            {
                "type": "image",
                "data": base64.b64encode(image).decode("utf-8"),
                "mime_type": sniff_image_mime(image),
            }
            for image in reference_images
        ]
        input_content.append({"type": "text", "text": prompt})

        # aspect_ratio works in both Developer API and Enterprise/Vertex mode.
        # mime_type in response_format is only ever "image/jpeg" in this SDK
        # version (there is no way to explicitly request PNG) -- omit it
        # unless jpeg is actually wanted, and let the API's default apply.
        response_format: dict[str, Any] = {
            "type": "image",
            "aspect_ratio": params.get("aspect_ratio", self.aspect_ratio),
        }
        if self.output_mime_type == "image/jpeg":
            response_format["mime_type"] = "image/jpeg"

        tools: list[dict[str, Any]] = []
        if self._image_search_enabled:
            tools.append({"type": "google_search", "search_types": ["web_search", "image_search"]})

        interaction = self.client.interactions.create(
            model=self.model_id,
            input=input_content,
            previous_interaction_id=params.get("previous_interaction_id"),
            # response_modalities MUST be IMAGE-only, not ["TEXT", "IMAGE"]:
            # these models will happily return a prose *description* of the
            # image instead of drawing it if TEXT is an allowed response
            # modality -- especially on elaborate prompts with reference
            # images. Omitting TEXT removes that escape hatch and reliably
            # yields image bytes.
            response_modalities=["IMAGE"],
            response_format=response_format,
            tools=tools or None,
        )

        output_image = interaction.output_image
        if output_image is not None and output_image.data is not None:
            return ImageResult(
                image_bytes=base64.b64decode(output_image.data),
                mime_type=output_image.mime_type or self.output_mime_type,
                provider="gemini",
                model_id=self.model_id,
                interaction_id=interaction.id,
            )
        raise RuntimeError(_describe_missing_image(interaction))
