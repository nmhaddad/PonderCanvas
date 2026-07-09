from collections.abc import Callable

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models import BaseLlm
from google.adk.tools import ToolContext

from pondercanvas.agent import prompts

GenerationTool = Callable[[ToolContext], dict]
EvaluationTool = Callable[[ToolContext], dict]
ExitLoopTool = Callable[[ToolContext], dict]


def build_generation_agent(chat_model: BaseLlm, generation_tool: GenerationTool) -> LlmAgent:
    return LlmAgent(
        name="GenerationAgent",
        model=chat_model,
        tools=[generation_tool],
        instruction=prompts.GENERATION_INSTRUCTION,
    )


def build_evaluation_agent(chat_model: BaseLlm, evaluation_tool: EvaluationTool) -> LlmAgent:
    return LlmAgent(
        name="EvaluationAgent",
        model=chat_model,
        tools=[evaluation_tool],
        instruction=prompts.EVALUATION_INSTRUCTION,
    )


def build_loop_control_agent(chat_model: BaseLlm, exit_loop_tool: ExitLoopTool) -> LlmAgent:
    return LlmAgent(
        name="LoopControlAgent",
        model=chat_model,
        tools=[exit_loop_tool],
        instruction=prompts.LOOP_CONTROL_INSTRUCTION,
    )


def build_refinement_loop(sub_agents: list, max_iterations: int) -> LoopAgent:
    """google.adk.agents.LoopAgent is deprecated in favor of a new Workflow
    primitive as of google-adk 2.4.0, but Workflow does not yet support
    being composed the way this pipeline needs (confirmed via spike against
    the installed version); LoopAgent remains the documented, working
    generate-critique-refine primitive. Revisit if/when Workflow matures."""
    return LoopAgent(name="RefinementLoop", sub_agents=sub_agents, max_iterations=max_iterations)
