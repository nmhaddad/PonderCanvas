from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None


class PhotoAttribution(BaseModel):
    """Credit for a downloaded Unsplash reference photo. photographer_name/
    photographer_profile_url + the Unsplash homepage link (rendered
    alongside this) are the credit Unsplash's API guidelines require;
    photo_page_url is an extra, non-required link to the photo itself."""

    photographer_name: str
    photographer_profile_url: str
    photo_page_url: str


class GroundingResult(BaseModel):
    """Output of the pre-loop grounding step: Gemini Google Search text
    grounding plus citations, and any real reference photos downloaded from
    Unsplash alongside it."""

    queries_used: list[str] = Field(default_factory=list)
    summary_text: str = ""
    citations: list[SourceCitation] = Field(default_factory=list)
    downloaded_reference_count: int = 0
    photo_attributions: list[PhotoAttribution] = Field(default_factory=list)
