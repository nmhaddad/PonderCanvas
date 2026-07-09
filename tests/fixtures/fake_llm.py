import re
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
    with LoopAgent/LlmAgent."""

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


class ControlDecisionFakeLlm(BaseLlm):
    """Scripted for the LoopControlAgent role: scans the conversation history
    fed back to it for the most recent evaluate_image result and calls
    exit_loop only if that result's "pass" field is true. Cross-agent tool
    results arrive as descriptive text (e.g. "`evaluate_image` tool returned
    result: {'pass': True, ...}"), not structured function_response parts --
    ADK flattens other agents' tool calls to text summaries for context, the
    same way a real model would read them. This exercises real cross-agent
    state propagation through ADK's conversation history, not a shortcut
    around it."""

    watch_tool_name: str = "evaluate_image"
    _awaiting_final_text: bool = False
    _result_pattern: re.Pattern = re.compile(r"tool returned result: (\{.*\})")

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if self._awaiting_final_text:
            self._awaiting_final_text = False
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="ok")]),
                turn_complete=True,
            )
            return

        passed = False
        for content in llm_request.contents or []:
            for part in content.parts or []:
                text = getattr(part, "text", None)
                if not text or self.watch_tool_name not in text:
                    continue
                match = self._result_pattern.search(text)
                if match:
                    passed = "'pass': True" in match.group(1)

        if passed:
            part = types.Part(function_call=types.FunctionCall(name="exit_loop", args={}))
            self._awaiting_final_text = True
            yield LlmResponse(content=types.Content(role="model", parts=[part]))
        else:
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="not yet")]),
                turn_complete=True,
            )


class PipelineFakeLlm(BaseLlm):
    """One model instance shared across GenerationAgent/EvaluationAgent/
    LoopControlAgent, exactly as the real pipeline does with a single
    build_chat_model() result. Dispatches behavior based on which tool the
    calling agent was given: unconditionally calls generate_image or
    evaluate_image; for exit_loop, reuses ControlDecisionFakeLlm's
    text-scanning decision logic. Used by full end-to-end pipeline tests."""

    _awaiting_final_text: bool = False
    _result_pattern: re.Pattern = re.compile(r"tool returned result: (\{.*\})")

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

        if "exit_loop" in tools_dict:
            passed = False
            for content in llm_request.contents or []:
                for part in content.parts or []:
                    text = getattr(part, "text", None)
                    if not text or "evaluate_image" not in text:
                        continue
                    match = self._result_pattern.search(text)
                    if match:
                        passed = "'pass': True" in match.group(1)
            if passed:
                part = types.Part(function_call=types.FunctionCall(name="exit_loop", args={}))
                self._awaiting_final_text = True
                yield LlmResponse(content=types.Content(role="model", parts=[part]))
            else:
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text="not yet")]),
                    turn_complete=True,
                )
            return

        tool_name = next(iter(tools_dict))
        part = types.Part(function_call=types.FunctionCall(name=tool_name, args={}))
        self._awaiting_final_text = True
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class TextOnlyFakeLlm(BaseLlm):
    """Scripted BaseLlm double that never calls a tool -- just replies with
    fixed text. Useful for negative-path tests (e.g. an agent that should
    not call exit_loop when the evaluation hasn't passed)."""

    reply_text: str = "ok"

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.reply_text)]),
            turn_complete=True,
        )
