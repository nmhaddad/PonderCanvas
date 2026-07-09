"""Deterministic graph nodes for the thinking-mode refinement Workflow
(see agent/workflow.py). Distinct from agent/tools/: a tool is only ever
called *by* an LLM agent via function-calling; a node here is wired directly
into the graph's edges and is never LLM-invoked."""

from collections.abc import Callable

from google.adk.agents import Context
from google.adk.events.event import Event, EventActions

from pondercanvas.agent import state_keys as sk

CheckStopConditionNode = Callable[[Context], Event]


def build_check_stop_condition_node(max_iterations: int) -> CheckStopConditionNode:
    def check_stop_condition(ctx: Context) -> Event:
        """Routes the refinement Workflow: "continue" loops back to
        GenerationAgent, "stop" ends the run. Reads the most recent
        evaluation and iteration count straight from session state -- the
        same deterministic check run_fast_refinement uses -- so this needs
        no chat-model call. When it routes "stop", the graph has no matching
        outgoing edge for that route and ends the branch there; ADK logs a
        benign "branch will end" warning on every normal completion, which
        is expected."""
        state = ctx.state
        last_evaluation = state.get(sk.LAST_EVALUATION) or {}
        iterations = state.get(sk.ITERATIONS, [])
        if last_evaluation.get("pass") or len(iterations) >= max_iterations:
            return Event(actions=EventActions(route="stop"))
        return Event(actions=EventActions(route="continue"))

    return check_stop_condition
