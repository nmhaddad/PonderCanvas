from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.nodes import build_check_stop_condition_node
from tests.fixtures.fake_tool_context import FakeToolContext


class TestCheckStopConditionNode:
    def test_continues_when_not_passing_and_under_cap(self):
        node = build_check_stop_condition_node(max_iterations=5)
        ctx = FakeToolContext(
            {sk.LAST_EVALUATION: {"pass": False}, sk.ITERATIONS: [{}]}
        )

        event = node(ctx)

        assert event.actions.route == "continue"

    def test_stops_when_evaluation_passed(self):
        node = build_check_stop_condition_node(max_iterations=5)
        ctx = FakeToolContext(
            {sk.LAST_EVALUATION: {"pass": True}, sk.ITERATIONS: [{}]}
        )

        event = node(ctx)

        assert event.actions.route == "stop"

    def test_stops_once_iteration_cap_is_reached_even_if_never_passing(self):
        node = build_check_stop_condition_node(max_iterations=2)
        ctx = FakeToolContext(
            {sk.LAST_EVALUATION: {"pass": False}, sk.ITERATIONS: [{}, {}]}
        )

        event = node(ctx)

        assert event.actions.route == "stop"

    def test_continues_when_no_evaluation_has_run_yet(self):
        # First iteration: no sk.LAST_EVALUATION key in state at all.
        node = build_check_stop_condition_node(max_iterations=5)
        ctx = FakeToolContext({sk.ITERATIONS: []})

        event = node(ctx)

        assert event.actions.route == "continue"

    def test_each_call_gets_a_fresh_iteration_cap_from_the_factory(self):
        strict_node = build_check_stop_condition_node(max_iterations=1)
        lenient_node = build_check_stop_condition_node(max_iterations=5)
        ctx = FakeToolContext({sk.LAST_EVALUATION: {"pass": False}, sk.ITERATIONS: [{}]})

        assert strict_node(ctx).actions.route == "stop"
        assert lenient_node(ctx).actions.route == "continue"
