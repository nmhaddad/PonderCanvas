from pydantic import BaseModel, Field


class GenerationBrief(BaseModel):
    """Structured intent extracted from the user's prompt + reference images."""

    subject: str
    style: str
    composition: str
    mood: str
    palette: str
    constraints: list[str] = Field(default_factory=list)
    notes_from_references: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    aspect_ratio: str = "1:1"
    raw_user_prompt: str
