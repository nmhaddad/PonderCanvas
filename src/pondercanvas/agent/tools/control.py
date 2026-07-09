from google.adk.tools import ToolContext


def exit_loop(tool_context: ToolContext) -> dict:
    """Ends the enclosing LoopAgent early. Call only when the most recent
    evaluation passed."""
    tool_context.actions.escalate = True
    return {"status": "loop_exited"}
