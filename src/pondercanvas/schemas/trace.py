from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from pondercanvas.schemas.brief import GenerationBrief
from pondercanvas.schemas.evaluation import EvaluationResult
from pondercanvas.schemas.grounding import GroundingResult


class IterationTrace(BaseModel):
    iteration_index: int
    prompt_used: str
    image_path: str
    evaluation: EvaluationResult | None = None
    created_at: datetime


class RunTrace(BaseModel):
    run_id: str
    brief: GenerationBrief
    grounding: GroundingResult | None = None
    iterations: list[IterationTrace] = Field(default_factory=list)
    final_image_path: str | None = None
    passed: bool = False
    stopped_reason: Literal["passed", "max_iterations_reached", "error"]
    settings_snapshot: dict = Field(default_factory=dict)
    created_at: datetime
