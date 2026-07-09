import uuid
from datetime import UTC, datetime

from google.adk.runners import InMemoryRunner
from google.genai import types

from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.agents import (
    build_evaluation_agent,
    build_generation_agent,
    build_loop_control_agent,
    build_refinement_loop,
)
from pondercanvas.agent.extraction import extract_generation_brief
from pondercanvas.agent.tools.control import exit_loop
from pondercanvas.agent.tools.evaluation_tool import make_evaluate_image_tool
from pondercanvas.agent.tools.generation_tool import make_generate_image_tool
from pondercanvas.config.settings import EffectiveSettings
from pondercanvas.logging_utils import log_run_trace
from pondercanvas.providers.chat.factory import build_chat_model
from pondercanvas.providers.image.registry import get_image_provider
from pondercanvas.providers.scoring.siglip import SiglipScorer
from pondercanvas.providers.search.collect import collect_references
from pondercanvas.providers.structured.gemini_structured import GeminiStructuredVisionProvider
from pondercanvas.schemas.grounding import GroundingResult
from pondercanvas.schemas.trace import IterationTrace, RunTrace

_IMAGE_PROVIDER_KEY_FIELD = {
    "gemini": "gemini_image_api_key",
    "openai": "openai_api_key",
    "stability": "stability_api_key",
}

_APP_NAME = "pondercanvas"
_USER_ID = "local"


class PonderCanvasPipeline:
    """Top-level orchestrator: extraction and reference-gathering run once as
    plain pre-steps, then ADK's LoopAgent drives generate -> evaluate ->
    control up to settings.max_iterations."""

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

        chat_model = build_chat_model(settings)
        loop = build_refinement_loop(
            [
                build_generation_agent(chat_model, generation_tool),
                build_evaluation_agent(chat_model, evaluation_tool),
                build_loop_control_agent(chat_model, exit_loop),
            ],
            max_iterations=settings.max_iterations,
        )

        runner = InMemoryRunner(agent=loop, app_name=_APP_NAME)
        session = await runner.session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            state={
                sk.BRIEF: brief.model_dump(),
                sk.GROUNDING_RESULT: grounding.model_dump(),
                sk.REFERENCE_IMAGE_BYTES: all_reference_images,
            },
        )
        async for _event in runner.run_async(
            user_id=_USER_ID,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            pass

        final_session = await runner.session_service.get_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session.id
        )
        if final_session is None:
            raise RuntimeError(f"Session {session.id!r} disappeared during pipeline run")
        trace = _assemble_run_trace(brief, grounding, final_session.state, settings)
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

    return RunTrace(
        run_id=str(uuid.uuid4()),
        brief=brief,
        grounding=grounding,
        iterations=iterations,
        final_image_path=state.get(sk.LAST_IMAGE_PATH),
        passed=passed,
        stopped_reason="passed" if passed else "max_iterations_reached",
        settings_snapshot=settings.redacted(),
        created_at=datetime.now(UTC),
    )
