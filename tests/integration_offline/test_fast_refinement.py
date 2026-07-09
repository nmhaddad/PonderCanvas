from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.refinement import run_fast_refinement
from pondercanvas.agent.tools.evaluation_tool import make_evaluate_image_tool
from pondercanvas.agent.tools.generation_tool import make_generate_image_tool
from pondercanvas.schemas.evaluation import EvaluationResult
from tests.fixtures.fake_image_provider import FakeImageProvider
from tests.fixtures.fake_structured_provider import FakeStructuredVisionProvider
from tests.fixtures.sample_brief import sample_brief


def _scores(value: float) -> dict:
    return {
        "prompt_adherence": value,
        "aesthetic_quality": value,
        "technical_quality": value,
        "reference_alignment": value,
    }


def _failing_eval(threshold=4.0) -> EvaluationResult:
    return EvaluationResult(
        scores=_scores(2.0), overall=2.0, is_passing=False, feedback="try again", threshold=threshold
    )


def _passing_eval(threshold=4.0) -> EvaluationResult:
    return EvaluationResult(
        scores=_scores(5.0), overall=5.0, is_passing=True, feedback="great", threshold=threshold
    )


def _tools(tmp_path, eval_results):
    image_provider = FakeImageProvider()
    structured_provider = FakeStructuredVisionProvider(eval_results)
    generation_tool = make_generate_image_tool(image_provider, tmp_path)
    evaluation_tool = make_evaluate_image_tool(structured_provider, threshold=4.0)
    return image_provider, structured_provider, generation_tool, evaluation_tool


class TestFastRefinementRunsToMaxIterationsWhenNeverPassing:
    def test_generation_called_exactly_max_iterations_times(self, tmp_path):
        image_provider, structured_provider, gen, ev = _tools(tmp_path, [_failing_eval()] * 10)

        final_state = run_fast_refinement(
            gen, ev, {sk.BRIEF: sample_brief().model_dump()}, max_iterations=5
        )

        assert len(image_provider.calls) == 5
        assert len(structured_provider.calls) == 5
        assert len(final_state[sk.ITERATIONS]) == 5
        assert final_state[sk.LAST_EVALUATION]["pass"] is False


class TestFastRefinementExitsEarlyOnPass:
    def test_stops_as_soon_as_evaluation_passes(self, tmp_path):
        # Fails twice, then passes on the 3rd iteration.
        image_provider, structured_provider, gen, ev = _tools(
            tmp_path, [_failing_eval(), _failing_eval(), _passing_eval()]
        )

        final_state = run_fast_refinement(
            gen, ev, {sk.BRIEF: sample_brief().model_dump()}, max_iterations=5
        )

        assert len(image_provider.calls) == 3
        assert len(structured_provider.calls) == 3
        assert final_state[sk.LAST_EVALUATION]["pass"] is True

    def test_passes_on_first_iteration_runs_only_once(self, tmp_path):
        image_provider, _, gen, ev = _tools(tmp_path, [_passing_eval()])

        run_fast_refinement(gen, ev, {sk.BRIEF: sample_brief().model_dump()}, max_iterations=5)

        assert len(image_provider.calls) == 1

    def test_does_not_mutate_the_provided_initial_state(self, tmp_path):
        _, _, gen, ev = _tools(tmp_path, [_passing_eval()])
        initial_state = {sk.BRIEF: sample_brief().model_dump()}

        run_fast_refinement(gen, ev, initial_state, max_iterations=5)

        assert sk.ITERATIONS not in initial_state
        assert sk.LAST_EVALUATION not in initial_state
