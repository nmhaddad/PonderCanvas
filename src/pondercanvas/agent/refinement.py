"""Interchangeable strategies for turning a prepared brief into a final
image. All three consume the same generate/evaluate tools and leave
identical session state behind, so downstream trace assembly is mode-agnostic.

"thinking" drives generate -> evaluate -> repeat through a google.adk.workflow
Workflow graph (see agent/workflow.py): one real chat-model round-trip per
LlmAgent per iteration, plus a deterministic check_stop_condition node
(agent/nodes.py) that routes the loop with no model call. Its generation
step composes its own prompt (rather than the fixed template fast/instant
use) and can optionally call search_reference_images/search_web when it
decides it needs more context -- see prompts/templates/generation_instruction.md
and tools/research.py.

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

import weave
from google.adk.models import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.agents import build_evaluation_agent, build_generation_agent
from pondercanvas.agent.nodes import build_check_stop_condition_node
from pondercanvas.agent.workflow import build_refinement_workflow

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


@weave.op()
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


@weave.op()
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


@weave.op()
async def run_thinking_refinement(
    chat_model: BaseLlm,
    generation_tool: RefinementTool,
    evaluation_tool: RefinementTool,
    initial_state: dict,
    max_iterations: int,
    prompt: str,
    *research_tools: RefinementTool,
) -> dict:
    """Drive generate -> evaluate -> repeat through a Workflow graph. Returns
    the final session state dict. `research_tools` (search_reference_images /
    search_web, if configured) are handed to the generation agent, which
    decides for itself each turn whether it needs to call them."""
    workflow = build_refinement_workflow(
        build_generation_agent(chat_model, generation_tool, *research_tools),
        build_evaluation_agent(chat_model, evaluation_tool),
        build_check_stop_condition_node(max_iterations),
    )

    # InMemoryRunner's stub still types `agent` as BaseAgent; Workflow (a
    # BaseNode) is a legitimate root agent as of google-adk 2.4.0's graph
    # support -- confirmed working end-to-end, the stub just hasn't caught up.
    runner = InMemoryRunner(agent=workflow, app_name=_APP_NAME)  # type: ignore[arg-type]
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
