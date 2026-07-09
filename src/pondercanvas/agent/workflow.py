"""Builds the thinking-mode refinement Workflow: a graph of
GenerationAgent -> EvaluationAgent -> check_stop_condition, looping back to
GenerationAgent until check_stop_condition routes to "stop". Replaces the
deprecated LoopAgent primitive (google.adk.agents.LoopAgent) now that
google-adk 2.4.0's Workflow graph supports everything this loop needs:
LlmAgent nodes, and conditional (routed) edges to express the cycle -- see
agent/agents.py for the two LlmAgents and agent/nodes.py for the
stop-condition node."""

from google.adk.agents import LlmAgent
from google.adk.workflow import START, Workflow

from pondercanvas.agent.nodes import CheckStopConditionNode


def build_refinement_workflow(
    generation_agent: LlmAgent,
    evaluation_agent: LlmAgent,
    check_stop_condition: CheckStopConditionNode,
) -> Workflow:
    return Workflow(
        name="RefinementWorkflow",
        edges=[
            (START, generation_agent, evaluation_agent, check_stop_condition),
            (check_stop_condition, {"continue": generation_agent}),
        ],
    )
