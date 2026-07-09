from google.adk.agents import LlmAgent, LoopAgent

from pondercanvas.agent.agents import (
    build_evaluation_agent,
    build_generation_agent,
    build_loop_control_agent,
    build_refinement_loop,
)
from pondercanvas.agent.tools.control import exit_loop
from tests.fixtures.fake_llm import ToolCallingFakeLlm


def _noop_tool(tool_context):
    return {}


class TestAgentConstruction:
    def test_generation_agent_wraps_given_tool(self):
        agent = build_generation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        assert isinstance(agent, LlmAgent)
        assert agent.name == "GenerationAgent"
        assert len(agent.tools) == 1

    def test_evaluation_agent_wraps_given_tool(self):
        agent = build_evaluation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        assert isinstance(agent, LlmAgent)
        assert agent.name == "EvaluationAgent"

    def test_loop_control_agent_wraps_exit_loop(self):
        agent = build_loop_control_agent(ToolCallingFakeLlm(model="fake"), exit_loop)
        assert isinstance(agent, LlmAgent)
        assert agent.name == "LoopControlAgent"
        assert agent.tools == [exit_loop]

    def test_refinement_loop_wires_sub_agents_in_order(self):
        gen = build_generation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        ev = build_evaluation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        control = build_loop_control_agent(ToolCallingFakeLlm(model="fake"), exit_loop)

        loop = build_refinement_loop([gen, ev, control], max_iterations=5)

        assert isinstance(loop, LoopAgent)
        assert [a.name for a in loop.sub_agents] == [
            "GenerationAgent",
            "EvaluationAgent",
            "LoopControlAgent",
        ]

    def test_refinement_loop_max_iterations_is_wired(self):
        gen = build_generation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
        loop = build_refinement_loop([gen], max_iterations=3)
        assert loop.max_iterations == 3
