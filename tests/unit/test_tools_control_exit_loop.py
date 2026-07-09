from pondercanvas.agent.tools.control import exit_loop
from tests.fixtures.fake_tool_context import FakeToolContext


class TestExitLoop:
    def test_sets_escalate_true(self):
        ctx = FakeToolContext()
        exit_loop(ctx)
        assert ctx.actions.escalate is True

    def test_returns_status_dict(self):
        ctx = FakeToolContext()
        result = exit_loop(ctx)
        assert result == {"status": "loop_exited"}

    def test_escalate_defaults_to_none_before_call(self):
        ctx = FakeToolContext()
        assert ctx.actions.escalate is None
