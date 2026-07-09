from google.adk.agents import LlmAgent

from pondercanvas.agent.agents import build_evaluation_agent, build_generation_agent
from tests.fixtures.fake_llm import ToolCallingFakeLlm


def _noop_tool(tool_context):
    return {}


class TestAgentConstruction:
    def test_generation_agent_wraps_given_tool(self):
        agent = build_generation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        assert isinstance(agent, LlmAgent)
        assert agent.name == "GenerationAgent"
        assert len(agent.tools) == 1

    def test_generation_agent_includes_research_tools_when_given(self):
        agent = build_generation_agent(
            ToolCallingFakeLlm(model="fake"), _noop_tool, _noop_tool, _noop_tool
        )
        assert len(agent.tools) == 3

    def test_generation_agent_omits_research_tools_when_not_given(self):
        agent = build_generation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        assert agent.tools == [_noop_tool]

    def test_evaluation_agent_wraps_given_tool(self):
        agent = build_evaluation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        assert isinstance(agent, LlmAgent)
        assert agent.name == "EvaluationAgent"
        assert agent.tools == [_noop_tool]
