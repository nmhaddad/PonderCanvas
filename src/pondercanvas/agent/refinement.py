"""Interchangeable strategies for turning a prepared brief into a final
image. All three consume the same generate/evaluate tools and leave
identical session state behind, so downstream trace assembly is mode-agnostic.

"thinking" drives generate -> evaluate -> control through ADK's LoopAgent:
one real chat-model round-trip per sub-agent per iteration. It's the place
richer, LLM-mediated reasoning will grow.

"fast" runs the same two tools in a plain Python for-loop and reads the
stop/continue decision straight from evaluation state
(`last_evaluation["pass"]`, already computed in `evaluate_image`). That
deterministic check needs no model, so fast spends zero LLM calls on
orchestration -- see GitHub issue #4.

"instant" skips the loop entirely: a single generate call, no evaluation.
For callers who just want one image out of the preloop work (extraction +
grounding) with no refinement spend at all.
"""

from collections.abc import Callable
from typing import Any

from google.adk.models import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.agents import (
    build_evaluation_agent,
    build_generation_agent,
    build_loop_control_agent,
    build_refinement_loop,
)
from pondercanvas.agent.tools.control import exit_loop

# The generate/evaluate tools are typed against ADK's ToolContext; the fast
# path feeds them a duck-typed stand-in, so accept Any at this boundary.
RefinementTool = Callable[[Any], dict]

_APP_NAME = "pondercanvas"
_USER_ID = "local"


class _FastActions:
    def __init__(self) -> None:
        self.escalate: bool | None = None


class _FastToolContext:
    """Minimal ToolContext stand-in for the fast for-loop path. The generate
    and evaluate tools only ever touch `.state`; the loop reads the stop
    decision directly from that dict, so no invocation/LLM plumbing is needed.
    `.actions` is present only so the same tools stay drop-in compatible."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.actions = _FastActions()


def run_fast_refinement(
    generation_tool: RefinementTool,
    evaluation_tool: RefinementTool,
    initial_state: dict,
    max_iterations: int,
) -> dict:
    """Generate then evaluate up to `max_iterations` times, stopping as soon as
    an evaluation passes. Returns the final state dict."""
    state = dict(initial_state)
    context = _FastToolContext(state)
    for _ in range(max_iterations):
        generation_tool(context)
        evaluation_tool(context)
        last_evaluation = state.get(sk.LAST_EVALUATION) or {}
        if last_evaluation.get("pass"):
            break
    return state


def run_instant_generation(
    generation_tool: RefinementTool,
    initial_state: dict,
) -> dict:
    """Run a single generation call with no evaluation and no loop. Returns
    the final state dict."""
    state = dict(initial_state)
    context = _FastToolContext(state)
    generation_tool(context)
    return state


async def run_thinking_refinement(
    chat_model: BaseLlm,
    generation_tool: RefinementTool,
    evaluation_tool: RefinementTool,
    initial_state: dict,
    max_iterations: int,
    prompt: str,
) -> dict:
    """Drive generate -> evaluate -> control through ADK's LoopAgent. Returns
    the final session state dict."""
    loop = build_refinement_loop(
        [
            build_generation_agent(chat_model, generation_tool),
            build_evaluation_agent(chat_model, evaluation_tool),
            build_loop_control_agent(chat_model, exit_loop),
        ],
        max_iterations=max_iterations,
    )

    runner = InMemoryRunner(agent=loop, app_name=_APP_NAME)
    session = await runner.session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID, state=initial_state
    )
    async for _event in runner.run_async(
        user_id=_USER_ID,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        pass

    final_session = await runner.session_service.get_session(
        app_name=_APP_NAME, user_id=_USER_ID, session_id=session.id
    )
    if final_session is None:
        raise RuntimeError(f"Session {session.id!r} disappeared during pipeline run")
    return final_session.state
