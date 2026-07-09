from pydantic import BaseModel, ConfigDict, Field


class CriterionScores(BaseModel):
    """Fixed set of per-criterion scores (see prompts/templates/eval_prompt.md.j2).
    Deliberately not a free-form dict[str, float]: Gemini's Developer API
    structured output rejects `additionalProperties` (the JSON Schema keyword
    Pydantic emits for arbitrary-keyed dict fields) -- only Gemini Enterprise
    Agent Platform mode supports that. siglip is optional and only populated
    when SigLIP scoring is blended in after the fact (see evaluation_tool.py)."""

    prompt_adherence: float
    aesthetic_quality: float
    technical_quality: float
    reference_alignment: float
    siglip: float | None = None


class EvaluationResult(BaseModel):
    """Structured critic output, including the "pass" alias (Gemini's
    structured JSON output uses the literal key "pass", which is a Python
    keyword)."""

    model_config = ConfigDict(populate_by_name=True)

    scores: CriterionScores
    overall: float
    is_passing: bool = Field(alias="pass")
    feedback: str
    threshold: float
