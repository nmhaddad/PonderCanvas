import weave
from pydantic import BaseModel, Field

from pondercanvas.agent.prompts import build_extraction_prompt
from pondercanvas.providers.structured.base import StructuredVisionProvider
from pondercanvas.schemas.brief import GenerationBrief


class _ExtractedFields(BaseModel):
    """What the LLM actually produces. raw_user_prompt and aspect_ratio are
    filled in by us afterward from known values, not trusted to the model's
    fidelity in echoing them back."""

    subject: str
    style: str
    composition: str
    mood: str
    palette: str
    constraints: list[str] = Field(default_factory=list)
    notes_from_references: str | None = None
    search_queries: list[str] = Field(default_factory=list)


@weave.op()
def extract_generation_brief(
    user_prompt: str,
    reference_images: list[bytes],
    structured_provider: StructuredVisionProvider,
    aspect_ratio: str = "1:1",
) -> GenerationBrief:
    prompt = build_extraction_prompt(user_prompt)
    extracted = structured_provider.generate_structured(prompt, reference_images, _ExtractedFields)
    return GenerationBrief(
        **extracted.model_dump(),
        aspect_ratio=aspect_ratio,
        raw_user_prompt=user_prompt,
    )
