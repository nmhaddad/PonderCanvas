import base64
from typing import Any, cast

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
            if self._enterprise:
                # enterprise=True routes through the Gemini Enterprise Agent
                # Platform (formerly Vertex AI) endpoint. Confirmed live
                # (2026-07-09, ADC auth, project 447368677805): the
                # interactions.create endpoint itself is now reachable there
                # (no 404, no permission error) but rejects every model tried
                # -- image and chat alike -- with a 400 "Unsupported model
                # interaction". So the Interactions API this provider relies
                # on still isn't usable via this endpoint for any model, just
                # for a different reason than before (a clean rejection, not
                # a gateway-level 404). Fail fast with an actionable message
                # instead of letting that surface from deep inside the SDK.
                raise RuntimeError(
                    "Gemini image generation via the Interactions API does not "
                    "currently work on the Enterprise/Vertex AI endpoint "
                    "(PONDERCANVAS_GEMINI_IMAGE_ENTERPRISE=true) -- every model is "
                    "rejected there with 'Unsupported model interaction'. Disable it "
                    "to use the standard Gemini Developer API endpoint instead."
                )
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
            # response_modalities MUST be image-only, not ["text", "image"]:
            # these models will happily return a prose *description* of the
            # image instead of drawing it if text is an allowed response
            # modality -- especially on elaborate prompts with reference
            # images. Omitting text removes that escape hatch and reliably
            # yields image bytes. Values are lowercase -- confirmed live
            # (2026-07-10): uppercase "IMAGE" now 400s with "not supported
            # for response_modalities[0]", even on the standard Developer
            # API, not just Vertex.
            response_modalities=["image"],
            response_format=response_format,
            tools=tools or None,
        )
        # .create() is typed to return Interaction | Stream[...] since the SDK
        # overloads on a `stream` kwarg we never pass -- always Interaction at
        # runtime here. Loosely-typed dict payloads above (vs the SDK's exact
        # TypedDicts) keep mypy from picking the narrower overload itself.
        # cast(), not isinstance(): a real runtime check would reject the
        # duck-typed fakes tests use in place of real SDK objects.
        interaction = cast(interactions.Interaction, interaction)

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
