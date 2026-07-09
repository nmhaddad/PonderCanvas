from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.tools.evaluation_tool import make_evaluate_image_tool
from pondercanvas.schemas.evaluation import EvaluationResult
from tests.fixtures.fake_siglip_scorer import FakeSiglipScorer
from tests.fixtures.fake_structured_provider import FakeStructuredVisionProvider
from tests.fixtures.fake_tool_context import FakeToolContext
from tests.fixtures.sample_brief import sample_brief


def _scores(value: float) -> dict:
    return {
        "prompt_adherence": value,
        "aesthetic_quality": value,
        "technical_quality": value,
        "reference_alignment": value,
    }


def _passing_result(threshold=4.0) -> EvaluationResult:
    return EvaluationResult(
        scores=_scores(5.0),
        overall=5.0,
        is_passing=True,
        feedback="Looks good",
        threshold=threshold,
    )


def _failing_result(threshold=4.0) -> EvaluationResult:
    return EvaluationResult(
        scores=_scores(2.0),
        overall=2.0,
        is_passing=False,
        feedback="bicycle is missing wheels",
        threshold=threshold,
    )


class TestEvaluateImageTool:
    def test_writes_last_evaluation_to_state(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        result = tool(ctx)

        assert result["pass"] is True
        assert result["overall"] == 5.0
        assert ctx.state[sk.LAST_EVALUATION]["pass"] is True

    def test_passes_generated_image_bytes_first(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        tool(ctx)

        assert provider.calls[0]["images"][0] == b"generated-bytes"

    def test_includes_reference_images_alongside_generated_image(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {
                sk.BRIEF: sample_brief().model_dump(),
                sk.LAST_IMAGE_PATH: str(image_path),
                sk.REFERENCE_IMAGE_BYTES: [b"ref1", b"ref2"],
            }
        )

        tool(ctx)

        assert provider.calls[0]["images"] == [b"generated-bytes", b"ref1", b"ref2"]

    def test_no_reference_images_evaluates_generated_image_alone(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        tool(ctx)

        assert provider.calls[0]["images"] == [b"generated-bytes"]

    def test_requests_evaluation_result_schema(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        tool(ctx)

        assert provider.calls[0]["response_schema"] is EvaluationResult

    def test_failing_evaluation_recorded_correctly(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_failing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        result = tool(ctx)

        assert result["pass"] is False
        assert ctx.state[sk.LAST_EVALUATION]["feedback"] == "bicycle is missing wheels"

    def test_attaches_evaluation_to_matching_iteration_trace_entry(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {
                sk.BRIEF: sample_brief().model_dump(),
                sk.LAST_IMAGE_PATH: str(image_path),
                sk.ITERATIONS: [
                    {"iteration_index": 0, "prompt_used": "p", "image_path": str(image_path)}
                ],
            }
        )

        tool(ctx)

        assert ctx.state[sk.ITERATIONS][0]["evaluation"]["pass"] is True

    def test_no_iterations_list_does_not_raise(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        tool(ctx)  # should not raise even though sk.ITERATIONS was never set


class TestEvaluateImageToolWithSiglip:
    def test_blends_siglip_score_into_overall_and_recomputes_pass(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        gemini_result = EvaluationResult(
            scores=_scores(3.0),
            overall=3.0,
            is_passing=False,
            feedback="needs more detail",
            threshold=4.0,
        )
        provider = FakeStructuredVisionProvider([gemini_result])
        siglip_scorer = FakeSiglipScorer(next_score=1.0)
        tool = make_evaluate_image_tool(
            provider, threshold=4.0, siglip_scorer=siglip_scorer, siglip_weight=0.6
        )
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        result = tool(ctx)

        # overall = (1 - 0.6) * 3.0 + 0.6 * (1 + 4 * 1.0) = 1.2 + 3.0 = 4.2
        assert result["overall"] == 4.2
        assert result["pass"] is True  # Gemini alone said False; the blend flips it
        assert ctx.state[sk.LAST_EVALUATION]["scores"]["siglip"] == 1.0

    def test_siglip_scorer_receives_generated_image_and_a_brief_derived_prompt(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        siglip_scorer = FakeSiglipScorer(next_score=0.5)
        tool = make_evaluate_image_tool(
            provider, threshold=4.0, siglip_scorer=siglip_scorer, siglip_weight=0.3
        )
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        tool(ctx)

        assert siglip_scorer.calls[0]["image_bytes"] == b"generated-bytes"
        assert "red bicycle" in siglip_scorer.calls[0]["prompt"]

    def test_siglip_returning_none_falls_back_to_gemini_only_evaluation(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        siglip_scorer = FakeSiglipScorer(next_score=None)  # simulates a failed/unavailable model
        tool = make_evaluate_image_tool(
            provider, threshold=4.0, siglip_scorer=siglip_scorer, siglip_weight=0.9
        )
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        result = tool(ctx)

        assert result["overall"] == 5.0
        assert result["pass"] is True
        assert ctx.state[sk.LAST_EVALUATION]["scores"]["siglip"] is None

    def test_no_siglip_scorer_configured_behaves_exactly_as_before(self, tmp_path):
        image_path = tmp_path / "iteration_0.png"
        image_path.write_bytes(b"generated-bytes")
        provider = FakeStructuredVisionProvider([_passing_result()])
        tool = make_evaluate_image_tool(provider, threshold=4.0)
        ctx = FakeToolContext(
            {sk.BRIEF: sample_brief().model_dump(), sk.LAST_IMAGE_PATH: str(image_path)}
        )

        result = tool(ctx)

        assert result == {"pass": True, "overall": 5.0}
