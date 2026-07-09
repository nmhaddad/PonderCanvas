from collections.abc import Callable

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from google.adk.tools import ToolContext

from pondercanvas.agent import prompts

GenerationTool = Callable[..., dict]
EvaluationTool = Callable[[ToolContext], dict]


def build_generation_agent(
    chat_model: BaseLlm,
    generation_tool: GenerationTool,
    *research_tools: GenerationTool,
) -> LlmAgent:
    """`research_tools` (search_reference_images / search_web) are optional:
    the agent decides per-turn whether it needs them before calling
    generation_tool. See prompts/templates/generation_instruction.md."""
    return LlmAgent(
        name="GenerationAgent",
        model=chat_model,
        tools=[generation_tool, *research_tools],
        instruction=prompts.GENERATION_INSTRUCTION,
    )


def build_evaluation_agent(chat_model: BaseLlm, evaluation_tool: EvaluationTool) -> LlmAgent:
    return LlmAgent(
        name="EvaluationAgent",
        model=chat_model,
        tools=[evaluation_tool],
        instruction=prompts.EVALUATION_INSTRUCTION,
    )
