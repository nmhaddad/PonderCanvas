class FakeActions:
    def __init__(self) -> None:
        self.escalate: bool | None = None


class FakeToolContext:
    """Minimal stand-in for google.adk.agents.context.ToolContext: our tool
    functions only ever touch .state (dict-like get/set) and
    .actions.escalate, so a plain dict + a tiny actions shim is enough."""

    def __init__(self, state: dict | None = None) -> None:
        self.state: dict = state if state is not None else {}
        self.actions = FakeActions()
