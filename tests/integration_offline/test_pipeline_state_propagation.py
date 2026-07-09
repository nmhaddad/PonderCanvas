import pytest

from pondercanvas.agent import pipeline as pipeline_module
from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.extraction import _ExtractedFields
from pondercanvas.config.settings import AppSettings, resolve_settings
from pondercanvas.schemas.evaluation import EvaluationResult
from pondercanvas.schemas.grounding import GroundingResult
from pondercanvas.schemas.trace import RunTrace
from tests.fixtures.fake_image_provider import FakeImageProvider
from tests.fixtures.fake_llm import PipelineFakeLlm
from tests.fixtures.fake_structured_provider import FakeStructuredVisionProvider
from tests.fixtures.sample_brief import sample_brief


def _effective(tmp_path, **overrides):
    defaults = dict(output_dir=tmp_path, google_api_key="fake-key")
    defaults.update(overrides)
    return resolve_settings(AppSettings(_env_file=None, **defaults))  # type: ignore[call-arg]


def _extracted_fields(**overrides) -> _ExtractedFields:
    brief_dict = sample_brief().model_dump()
    brief_dict.pop("aspect_ratio")
    brief_dict.pop("raw_user_prompt")
    brief_dict.update(overrides)
    return _ExtractedFields(**brief_dict)


def _scores(value: float) -> dict:
    return {
        "prompt_adherence": value,
        "aesthetic_quality": value,
        "technical_quality": value,
        "reference_alignment": value,
    }


def _passing_eval(**overrides) -> EvaluationResult:
    defaults = dict(scores=_scores(5.0), overall=5.0, is_passing=True, feedback="great", threshold=4.0)
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def _failing_eval(**overrides) -> EvaluationResult:
    defaults = dict(scores=_scores(1.0), overall=1.0, is_passing=False, feedback="nope", threshold=4.0)
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def _patch_pipeline_providers(monkeypatch, *, eval_results, extracted=None, downloaded_images=None):
    structured_provider = FakeStructuredVisionProvider([extracted or _extracted_fields(), *eval_results])
    image_provider = FakeImageProvider()

    monkeypatch.setattr(
        pipeline_module,
        "GeminiStructuredVisionProvider",
        lambda model_id, api_key: structured_provider,
    )
    monkeypatch.setattr(pipeline_module, "get_image_provider", lambda name, **kwargs: image_provider)
    monkeypatch.setattr(
        pipeline_module,
        "collect_references",
        lambda brief, settings: (
            GroundingResult(summary_text="grounded ctx"),
            downloaded_images or [],
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "build_chat_model", lambda settings: PipelineFakeLlm(model="fake")
    )
    return structured_provider, image_provider


@pytest.mark.asyncio
class TestPipelineStatePropagation:
    async def test_run_returns_run_trace_with_iterations(self, tmp_path, monkeypatch):
        _patch_pipeline_providers(monkeypatch, eval_results=[_passing_eval()])

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path))
        trace = await pipeline.run("draw a red bicycle", [])

        assert isinstance(trace, RunTrace)
        assert trace.passed is True
        assert trace.stopped_reason == "passed"
        assert len(trace.iterations) == 1
        assert trace.final_image_path is not None

    async def test_thinking_mode_drives_workflow_graph_to_the_same_result(self, tmp_path, monkeypatch):
        # The "thinking" Workflow-graph path must leave the same state behind
        # as the default "fast" for-loop: fails twice, then passes on the 3rd.
        _patch_pipeline_providers(
            monkeypatch, eval_results=[_failing_eval(), _failing_eval(), _passing_eval()]
        )

        settings = _effective(tmp_path, refinement_mode="thinking")
        pipeline = pipeline_module.PonderCanvasPipeline(settings)
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.passed is True
        assert trace.stopped_reason == "passed"
        assert len(trace.iterations) == 3

    async def test_instant_mode_generates_a_single_image_with_no_evaluation(
        self, tmp_path, monkeypatch
    ):
        structured_provider, image_provider = _patch_pipeline_providers(
            monkeypatch, eval_results=[]
        )

        settings = _effective(tmp_path, refinement_mode="instant")
        pipeline = pipeline_module.PonderCanvasPipeline(settings)
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.passed is False
        assert trace.stopped_reason == "instant"
        assert len(trace.iterations) == 1
        assert trace.iterations[0].evaluation is None
        assert trace.final_image_path is not None
        # extract_generation_brief consumes one structured-provider call; the
        # instant path must not consume a second one for evaluation.
        assert len(structured_provider.calls) == 1
        assert len(image_provider.calls) == 1

    async def test_thinking_mode_attaches_only_search_web_when_no_unsplash_key(
        self, tmp_path, monkeypatch
    ):
        _patch_pipeline_providers(monkeypatch, eval_results=[_passing_eval()])
        captured: dict = {}

        async def fake_run_thinking_refinement(
            chat_model, generation_tool, evaluation_tool, initial_state, max_iterations, prompt,
            *research_tools,
        ):
            captured["count"] = len(research_tools)
            return {sk.LAST_IMAGE_PATH: "x", sk.ITERATIONS: [], sk.LAST_EVALUATION: {"pass": True}}

        monkeypatch.setattr(pipeline_module, "run_thinking_refinement", fake_run_thinking_refinement)

        # _effective() defaults to a google_api_key but no unsplash_api_key.
        settings = _effective(tmp_path, refinement_mode="thinking")
        pipeline = pipeline_module.PonderCanvasPipeline(settings)
        await pipeline.run("draw a red bicycle", [])

        assert captured["count"] == 1

    async def test_thinking_mode_attaches_both_research_tools_when_both_keys_configured(
        self, tmp_path, monkeypatch
    ):
        _patch_pipeline_providers(monkeypatch, eval_results=[_passing_eval()])
        captured: dict = {}

        async def fake_run_thinking_refinement(
            chat_model, generation_tool, evaluation_tool, initial_state, max_iterations, prompt,
            *research_tools,
        ):
            captured["count"] = len(research_tools)
            return {sk.LAST_IMAGE_PATH: "x", sk.ITERATIONS: [], sk.LAST_EVALUATION: {"pass": True}}

        monkeypatch.setattr(pipeline_module, "run_thinking_refinement", fake_run_thinking_refinement)

        settings = _effective(tmp_path, refinement_mode="thinking", unsplash_api_key="u-key")
        pipeline = pipeline_module.PonderCanvasPipeline(settings)
        await pipeline.run("draw a red bicycle", [])

        assert captured["count"] == 2

    async def test_fast_mode_does_not_build_a_chat_model(self, tmp_path, monkeypatch):
        _patch_pipeline_providers(monkeypatch, eval_results=[_passing_eval()])

        calls: list[object] = []
        monkeypatch.setattr(
            pipeline_module,
            "build_chat_model",
            lambda settings: calls.append(settings) or PipelineFakeLlm(model="fake"),
        )

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path))  # fast is default
        await pipeline.run("draw a red bicycle", [])

        assert calls == []

    async def test_grounding_is_recorded_in_trace(self, tmp_path, monkeypatch):
        _patch_pipeline_providers(monkeypatch, eval_results=[_passing_eval()])

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path))
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.grounding.summary_text == "grounded ctx"

    async def test_max_iterations_reached_when_never_passing(self, tmp_path, monkeypatch):
        _patch_pipeline_providers(monkeypatch, eval_results=[_failing_eval()] * 10)

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path, max_iterations=2))
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.passed is False
        assert trace.stopped_reason == "max_iterations_reached"
        assert len(trace.iterations) == 2

    async def test_final_image_is_the_best_scoring_iteration_not_the_last(self, tmp_path, monkeypatch):
        # First iteration scores higher than the ones after it, but the run
        # never passes -- the final image should be the best-scoring
        # attempt, not just whichever happened to run last.
        _patch_pipeline_providers(
            monkeypatch,
            eval_results=[
                _failing_eval(overall=3.5),
                _failing_eval(overall=2.0),
                _failing_eval(overall=1.0),
            ],
        )

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path, max_iterations=3))
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.passed is False
        assert len(trace.iterations) == 3
        assert trace.final_image_path == trace.iterations[0].image_path
        assert trace.final_image_path != trace.iterations[-1].image_path

    async def test_final_image_is_the_passing_iteration_when_one_passes(self, tmp_path, monkeypatch):
        # Passing stops the loop immediately, so the passing iteration is
        # always the last one recorded -- final image should be that one
        # even though an earlier failing attempt might have scored close.
        _patch_pipeline_providers(
            monkeypatch,
            eval_results=[_failing_eval(overall=3.9), _passing_eval(overall=4.0)],
        )

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path))
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.passed is True
        assert trace.final_image_path == trace.iterations[-1].image_path

    async def test_settings_snapshot_redacts_api_key(self, tmp_path, monkeypatch):
        _patch_pipeline_providers(monkeypatch, eval_results=[_passing_eval()])

        settings = _effective(tmp_path, google_api_key="super-secret")
        pipeline = pipeline_module.PonderCanvasPipeline(settings)
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.settings_snapshot["google_api_key"] == "***REDACTED***"

    async def test_iteration_prompts_are_recorded(self, tmp_path, monkeypatch):
        _patch_pipeline_providers(monkeypatch, eval_results=[_failing_eval(), _passing_eval()])

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path))
        trace = await pipeline.run("draw a red bicycle", [])

        assert len(trace.iterations) == 2
        assert trace.iterations[0].evaluation.is_passing is False
        assert trace.iterations[1].evaluation.is_passing is True

    async def test_downloaded_reference_images_reach_image_generation(self, tmp_path, monkeypatch):
        # Regression test for the collect_references() -> pipeline state ->
        # generate_image tool -> ImageProvider.generate() wiring: photos
        # downloaded via collect_references (e.g. from Unsplash) must reach
        # the actual generation call, alongside any user-uploaded images.
        _, image_provider = _patch_pipeline_providers(
            monkeypatch,
            eval_results=[_passing_eval()],
            downloaded_images=[b"unsplash-photo-1", b"unsplash-photo-2"],
        )

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path))
        await pipeline.run("draw a red bicycle", [b"user-uploaded-photo"])

        assert image_provider.calls[0]["reference_images"] == [
            b"user-uploaded-photo",
            b"unsplash-photo-1",
            b"unsplash-photo-2",
        ]

    async def test_no_downloaded_images_only_user_uploads_reach_generation(self, tmp_path, monkeypatch):
        _, image_provider = _patch_pipeline_providers(monkeypatch, eval_results=[_passing_eval()])

        pipeline = pipeline_module.PonderCanvasPipeline(_effective(tmp_path))
        await pipeline.run("draw a red bicycle", [b"user-uploaded-photo"])

        assert image_provider.calls[0]["reference_images"] == [b"user-uploaded-photo"]


class TestAssembleRunTraceMergesMidLoopAttributions:
    def test_photo_attributions_collected_mid_loop_are_folded_into_grounding(self, tmp_path):
        brief = sample_brief()
        grounding = GroundingResult(summary_text="preloop grounding")
        state = {
            sk.ITERATIONS: [],
            sk.PHOTO_ATTRIBUTIONS: [
                {
                    "photographer_name": "Alice",
                    "photographer_profile_url": "https://unsplash.com/@alice",
                    "photo_page_url": "https://unsplash.com/photos/p1",
                }
            ],
        }
        settings = _effective(tmp_path, refinement_mode="thinking")

        trace = pipeline_module._assemble_run_trace(brief, grounding, state, settings)

        assert len(trace.grounding.photo_attributions) == 1
        assert trace.grounding.photo_attributions[0].photographer_name == "Alice"
        # The original GroundingResult passed in must not be mutated in place.
        assert grounding.photo_attributions == []

    def test_no_mid_loop_attributions_leaves_grounding_untouched(self, tmp_path):
        brief = sample_brief()
        grounding = GroundingResult(summary_text="preloop grounding")
        state = {sk.ITERATIONS: []}
        settings = _effective(tmp_path)

        trace = pipeline_module._assemble_run_trace(brief, grounding, state, settings)

        assert trace.grounding is grounding
