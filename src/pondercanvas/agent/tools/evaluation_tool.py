from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from google.adk.tools import ToolContext

from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.prompts import build_eval_prompt
from pondercanvas.providers.structured.base import StructuredVisionProvider
from pondercanvas.schemas.evaluation import EvaluationResult

EvaluateImageTool = Callable[[ToolContext], dict]


class SupportsSiglipScore(Protocol):
    def score(self, image_bytes: bytes, prompt: str) -> float | None: ...


def _siglip_prompt(brief: dict) -> str:
    return ", ".join(filter(None, [brief.get("subject"), brief.get("style")]))


def _blend_siglip_score(
    evaluation_dict: dict,
    siglip_scorer: SupportsSiglipScore,
    siglip_weight: float,
    image_bytes: bytes,
    brief: dict,
    threshold: float,
) -> dict:
    siglip_score = siglip_scorer.score(image_bytes, _siglip_prompt(brief))
    if siglip_score is None:
        # Scorer failed to initialize (e.g. optional deps missing): fall back
        # to Gemini's evaluation alone, as if siglip_weight were 0.
        return evaluation_dict

    siglip_scaled = 1.0 + 4.0 * siglip_score  # map SigLIP's [0, 1] onto Gemini's 1-5 scale
    overall = (1 - siglip_weight) * evaluation_dict["overall"] + siglip_weight * siglip_scaled
    return {
        **evaluation_dict,
        "scores": {**evaluation_dict["scores"], "siglip": siglip_score},
        "overall": overall,
        "pass": overall >= threshold,
    }


def make_evaluate_image_tool(
    structured_provider: StructuredVisionProvider,
    threshold: float,
    siglip_scorer: SupportsSiglipScore | None = None,
    siglip_weight: float = 0.0,
) -> EvaluateImageTool:
    def evaluate_image(tool_context: ToolContext) -> dict:
        """Scores the most recently generated image against the brief and any
        reference images on prompt adherence, aesthetic quality, technical
        quality, and reference alignment, returning pass/fail with feedback."""
        state = tool_context.state
        brief = state[sk.BRIEF]
        generated_image_bytes = Path(state[sk.LAST_IMAGE_PATH]).read_bytes()

        # The generated image is scored *together with* the user's/grounding's
        # reference images (not in isolation) so the critic can judge
        # reference/style fidelity.
        reference_images = state.get(sk.REFERENCE_IMAGE_BYTES, [])
        images = [generated_image_bytes, *reference_images]

        prompt = build_eval_prompt(brief, threshold)
        evaluation = structured_provider.generate_structured(prompt, images, EvaluationResult)

        evaluation_dict = evaluation.model_dump(by_alias=True)
        if siglip_scorer is not None:
            evaluation_dict = _blend_siglip_score(
                evaluation_dict, siglip_scorer, siglip_weight, generated_image_bytes, brief, threshold
            )
        state[sk.LAST_EVALUATION] = evaluation_dict

        iterations = list(state.get(sk.ITERATIONS, []))
        if iterations:
            iterations[-1] = {**iterations[-1], "evaluation": evaluation_dict}
            state[sk.ITERATIONS] = iterations

        return {"pass": evaluation_dict["pass"], "overall": evaluation_dict["overall"]}

    return evaluate_image
