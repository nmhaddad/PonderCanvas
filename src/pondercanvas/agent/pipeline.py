import uuid
from datetime import UTC, datetime
from typing import Literal

from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.extraction import extract_generation_brief
from pondercanvas.agent.refinement import (
    run_fast_refinement,
    run_instant_generation,
    run_thinking_refinement,
)
from pondercanvas.agent.tools.evaluation_tool import make_evaluate_image_tool
from pondercanvas.agent.tools.generation_tool import make_generate_image_tool
from pondercanvas.agent.tools.research import make_search_reference_images_tool, make_search_web_tool
from pondercanvas.config.settings import EffectiveSettings
from pondercanvas.logging_utils import log_run_trace
from pondercanvas.providers.chat.factory import build_chat_model
from pondercanvas.providers.image.registry import get_image_provider
from pondercanvas.providers.scoring.siglip import SiglipScorer
from pondercanvas.providers.search.collect import collect_references
from pondercanvas.providers.structured.gemini_structured import GeminiStructuredVisionProvider
from pondercanvas.schemas.grounding import GroundingResult, PhotoAttribution
from pondercanvas.schemas.trace import IterationTrace, RunTrace

_IMAGE_PROVIDER_KEY_FIELD = {
    "gemini": "gemini_image_api_key",
    "openai": "openai_api_key",
    "stability": "stability_api_key",
}


class PonderCanvasPipeline:
    """Top-level orchestrator: extraction and reference-gathering run once as
    plain pre-steps, then the configured refinement mode drives image
    generation: "fast" (plain for-loop) or "thinking" (Workflow graph) run
    generate -> evaluate -> repeat up to settings.max_iterations, while
    "instant" skips the loop and generates a single image."""

    def __init__(self, settings: EffectiveSettings):
        self.settings = settings
        self.structured_provider = GeminiStructuredVisionProvider(
            model_id=settings.structured_model_id, api_key=settings.google_api_key
        )
        self.image_provider = get_image_provider(
            settings.image_provider,
            model_id=settings.image_model_id,
            api_key=self._image_api_key(),
            aspect_ratio=settings.aspect_ratio,
            enterprise=settings.gemini_image_enterprise,
        )
        self.siglip_scorer = SiglipScorer() if settings.siglip_enabled else None

    def _image_api_key(self) -> str | None:
        key_field = _IMAGE_PROVIDER_KEY_FIELD.get(self.settings.image_provider)
        return getattr(self.settings, key_field) if key_field else None

    async def run(self, prompt: str, reference_images: list[bytes]) -> RunTrace:
        settings = self.settings

        brief = extract_generation_brief(
            prompt, reference_images, self.structured_provider, aspect_ratio=settings.aspect_ratio
        )
        grounding, downloaded_images = collect_references(brief, settings)
        all_reference_images = [*reference_images, *downloaded_images]

        generation_tool = make_generate_image_tool(self.image_provider, settings.output_dir)
        evaluation_tool = make_evaluate_image_tool(
            self.structured_provider,
            threshold=settings.eval_pass_threshold,
            siglip_scorer=self.siglip_scorer,
            siglip_weight=settings.siglip_weight,
        )

        initial_state = {
            sk.BRIEF: brief.model_dump(),
            sk.GROUNDING_RESULT: grounding.model_dump(),
            sk.REFERENCE_IMAGE_BYTES: all_reference_images,
        }

        if settings.refinement_mode == "thinking":
            research_tools = []
            if settings.unsplash_api_key:
                research_tools.append(
                    make_search_reference_images_tool(
                        settings.unsplash_api_key,
                        settings.max_reference_downloads,
                        settings.max_download_bytes,
                        settings.download_timeout_s,
                    )
                )
            if settings.google_api_key:
                research_tools.append(
                    make_search_web_tool(settings.google_api_key, settings.structured_model_id)
                )

            final_state = await run_thinking_refinement(
                build_chat_model(settings),
                generation_tool,
                evaluation_tool,
                initial_state,
                settings.max_iterations,
                prompt,
                *research_tools,
            )
        elif settings.refinement_mode == "instant":
            final_state = run_instant_generation(generation_tool, initial_state)
        else:
            final_state = run_fast_refinement(
                generation_tool, evaluation_tool, initial_state, settings.max_iterations
            )

        trace = _assemble_run_trace(brief, grounding, final_state, settings)
        log_run_trace(trace, settings.output_dir / "runs.jsonl")
        return trace


def _assemble_run_trace(
    brief, grounding: GroundingResult, state: dict, settings: EffectiveSettings
) -> RunTrace:
    iterations = [
        IterationTrace(
            iteration_index=it["iteration_index"],
            prompt_used=it["prompt_used"],
            image_path=it["image_path"],
            evaluation=it.get("evaluation"),
            created_at=datetime.fromisoformat(it["created_at"]),
        )
        for it in state.get(sk.ITERATIONS, [])
    ]

    last_evaluation = state.get(sk.LAST_EVALUATION)
    passed = bool(last_evaluation and last_evaluation.get("pass"))

    # search_reference_images (thinking mode only) may have pulled in extra
    # Unsplash photos mid-loop -- those aren't part of the preloop `grounding`
    # object, so fold their attributions in here for the trace.
    extra_attributions = [
        PhotoAttribution(**attribution) for attribution in state.get(sk.PHOTO_ATTRIBUTIONS, [])
    ]
    if extra_attributions:
        grounding = grounding.model_copy(
            update={"photo_attributions": [*grounding.photo_attributions, *extra_attributions]}
        )

    stopped_reason: Literal["passed", "max_iterations_reached", "instant"]
    if settings.refinement_mode == "instant":
        stopped_reason = "instant"
    elif passed:
        stopped_reason = "passed"
    else:
        stopped_reason = "max_iterations_reached"

    return RunTrace(
        run_id=str(uuid.uuid4()),
        brief=brief,
        grounding=grounding,
        iterations=iterations,
        final_image_path=state.get(sk.LAST_IMAGE_PATH),
        passed=passed,
        stopped_reason=stopped_reason,
        settings_snapshot=settings.redacted(),
        created_at=datetime.now(UTC),
    )
