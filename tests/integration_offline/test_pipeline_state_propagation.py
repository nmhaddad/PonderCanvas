import pytest

from pondercanvas.agent import pipeline as pipeline_module
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

    async def test_thinking_mode_drives_loop_agent_to_the_same_result(self, tmp_path, monkeypatch):
        # The "thinking" LoopAgent path must leave the same state behind as the
        # default "fast" for-loop: fails twice, then passes on the 3rd.
        _patch_pipeline_providers(
            monkeypatch, eval_results=[_failing_eval(), _failing_eval(), _passing_eval()]
        )

        settings = _effective(tmp_path, refinement_mode="thinking")
        pipeline = pipeline_module.PonderCanvasPipeline(settings)
        trace = await pipeline.run("draw a red bicycle", [])

        assert trace.passed is True
        assert trace.stopped_reason == "passed"
        assert len(trace.iterations) == 3

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
