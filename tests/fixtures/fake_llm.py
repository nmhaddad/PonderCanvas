from collections.abc import AsyncGenerator

from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class ToolCallingFakeLlm(BaseLlm):
    """Scripted BaseLlm double for integration_offline tests: on its first
    turn calls the given tool_name (or the sole available tool if none is
    given), then on the following turn (once the tool result is fed back)
    ends with plain text -- mirrors how a real model behaves after a
    function response comes back. See tests/integration_offline for usage
    with Workflow/LlmAgent."""

    tool_name: str | None = None
    _awaiting_final_text: bool = False

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if self._awaiting_final_text:
            self._awaiting_final_text = False
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="done")]),
                turn_complete=True,
            )
            return

        tools_dict = llm_request.tools_dict or {}
        tool_name = self.tool_name or next(iter(tools_dict))
        part = types.Part(function_call=types.FunctionCall(name=tool_name, args={}))
        self._awaiting_final_text = True
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class PipelineFakeLlm(BaseLlm):
    """One model instance shared across GenerationAgent/EvaluationAgent,
    exactly as the real pipeline does with a single build_chat_model()
    result. Unconditionally calls whichever tool the calling agent was
    given -- there's only one real choice per agent (generate_image or
    evaluate_image); research tools, if attached, are simply never called by
    this double. Continue/stop is decided by the deterministic
    check_stop_condition graph node, not the model, so this double doesn't
    need to reason about it. Used by full end-to-end pipeline tests."""

    _awaiting_final_text: bool = False

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if self._awaiting_final_text:
            self._awaiting_final_text = False
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="done")]),
                turn_complete=True,
            )
            return

        tools_dict = llm_request.tools_dict or {}
        tool_name = next(iter(tools_dict))
        part = types.Part(function_call=types.FunctionCall(name=tool_name, args={}))
        self._awaiting_final_text = True
        yield LlmResponse(content=types.Content(role="model", parts=[part]))
