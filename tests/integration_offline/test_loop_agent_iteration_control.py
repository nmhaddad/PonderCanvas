import pytest
from google.genai import types

from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.agents import (
    build_evaluation_agent,
    build_generation_agent,
    build_loop_control_agent,
    build_refinement_loop,
)
from pondercanvas.agent.tools.control import exit_loop
from pondercanvas.agent.tools.evaluation_tool import make_evaluate_image_tool
from pondercanvas.agent.tools.generation_tool import make_generate_image_tool
from pondercanvas.schemas.evaluation import EvaluationResult
from tests.fixtures.fake_image_provider import FakeImageProvider
from tests.fixtures.fake_llm import ControlDecisionFakeLlm, ToolCallingFakeLlm
from tests.fixtures.fake_structured_provider import FakeStructuredVisionProvider
from tests.fixtures.sample_brief import sample_brief


async def _run_loop(loop, initial_state):
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=loop, app_name="test-app")
    session = await runner.session_service.create_session(
        app_name="test-app", user_id="u", state=initial_state
    )
    async for _event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    ):
        pass
    return await runner.session_service.get_session(
        app_name="test-app", user_id="u", session_id=session.id
    )


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


@pytest.mark.asyncio
class TestLoopRunsToMaxIterationsWhenNeverPassing:
    async def test_generation_called_exactly_max_iterations_times(self, tmp_path):
        image_provider = FakeImageProvider()
        structured_provider = FakeStructuredVisionProvider([_failing_eval()] * 10)

        generation_tool = make_generate_image_tool(image_provider, tmp_path)
        evaluation_tool = make_evaluate_image_tool(structured_provider, threshold=4.0)

        gen_agent = build_generation_agent(ToolCallingFakeLlm(model="fake"), generation_tool)
        eval_agent = build_evaluation_agent(ToolCallingFakeLlm(model="fake"), evaluation_tool)
        # No control agent: nothing can ever call exit_loop.
        loop = build_refinement_loop([gen_agent, eval_agent], max_iterations=5)

        final_session = await _run_loop(
            loop, {sk.BRIEF: sample_brief().model_dump()}
        )

        assert len(image_provider.calls) == 5
        assert len(structured_provider.calls) == 5
        assert len(final_session.state[sk.ITERATIONS]) == 5


@pytest.mark.asyncio
class TestLoopExitsEarlyOnPass:
    async def test_stops_as_soon_as_evaluation_passes(self, tmp_path):
        image_provider = FakeImageProvider()
        # Fails twice, then passes on the 3rd iteration.
        structured_provider = FakeStructuredVisionProvider(
            [_failing_eval(), _failing_eval(), _passing_eval()]
        )

        generation_tool = make_generate_image_tool(image_provider, tmp_path)
        evaluation_tool = make_evaluate_image_tool(structured_provider, threshold=4.0)

        gen_agent = build_generation_agent(ToolCallingFakeLlm(model="fake"), generation_tool)
        eval_agent = build_evaluation_agent(ToolCallingFakeLlm(model="fake"), evaluation_tool)
        control_agent = build_loop_control_agent(ControlDecisionFakeLlm(model="fake"), exit_loop)
        loop = build_refinement_loop([gen_agent, eval_agent, control_agent], max_iterations=5)

        final_session = await _run_loop(
            loop, {sk.BRIEF: sample_brief().model_dump()}
        )

        assert len(image_provider.calls) == 3
        assert len(structured_provider.calls) == 3
        assert final_session.state[sk.LAST_EVALUATION]["pass"] is True

    async def test_never_exceeds_max_iterations_even_if_it_never_passes(self, tmp_path):
        image_provider = FakeImageProvider()
        structured_provider = FakeStructuredVisionProvider([_failing_eval()] * 10)

        generation_tool = make_generate_image_tool(image_provider, tmp_path)
        evaluation_tool = make_evaluate_image_tool(structured_provider, threshold=4.0)

        gen_agent = build_generation_agent(ToolCallingFakeLlm(model="fake"), generation_tool)
        eval_agent = build_evaluation_agent(ToolCallingFakeLlm(model="fake"), evaluation_tool)
        control_agent = build_loop_control_agent(ControlDecisionFakeLlm(model="fake"), exit_loop)
        loop = build_refinement_loop([gen_agent, eval_agent, control_agent], max_iterations=4)

        final_session = await _run_loop(
            loop, {sk.BRIEF: sample_brief().model_dump()}
        )

        assert len(image_provider.calls) == 4
        assert final_session.state[sk.LAST_EVALUATION]["pass"] is False
