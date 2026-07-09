from google.adk.workflow import START, Workflow

from pondercanvas.agent.agents import build_evaluation_agent, build_generation_agent
from pondercanvas.agent.nodes import build_check_stop_condition_node
from pondercanvas.agent.workflow import build_refinement_workflow
from tests.fixtures.fake_llm import ToolCallingFakeLlm


def _noop_tool(tool_context):
    return {}


def _build_workflow():
    gen = build_generation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
    ev = build_evaluation_agent(ToolCallingFakeLlm(model="fake"), _noop_tool)
    stop_check = build_check_stop_condition_node(max_iterations=3)
    return build_refinement_workflow(gen, ev, stop_check)


class TestBuildRefinementWorkflow:
    def test_returns_a_workflow(self):
        assert isinstance(_build_workflow(), Workflow)

    def test_wires_start_through_generation_evaluation_to_stop_check(self):
        workflow = _build_workflow()

        edges = {(e.from_node.name, e.to_node.name, e.route) for e in workflow.graph.edges}

        assert (START.name, "GenerationAgent", None) in edges
        assert ("GenerationAgent", "EvaluationAgent", None) in edges
        assert ("EvaluationAgent", "check_stop_condition", None) in edges

    def test_stop_check_routes_continue_back_to_generation_agent(self):
        workflow = _build_workflow()

        edges = {(e.from_node.name, e.to_node.name, e.route) for e in workflow.graph.edges}

        assert ("check_stop_condition", "GenerationAgent", "continue") in edges

    def test_stop_check_has_no_edge_for_the_stop_route(self):
        # "stop" intentionally has no matching edge: the graph ends the
        # branch there instead of routing anywhere further -- see
        # check_stop_condition's docstring in agent/nodes.py.
        workflow = _build_workflow()

        stop_check_routes = {
            e.route for e in workflow.graph.edges if e.from_node.name == "check_stop_condition"
        }

        assert stop_check_routes == {"continue"}
