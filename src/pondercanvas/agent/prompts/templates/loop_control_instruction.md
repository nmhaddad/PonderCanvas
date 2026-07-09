## Task
Decide whether to end the image-generation refinement loop.

## Context
The most recent `evaluate_image` tool result in the conversation contains a `pass` field.

## Instructions
- If `pass` is `true`: call the `exit_loop` tool immediately.
- If `pass` is `false`, or no `evaluate_image` result is visible yet: do not call
  any tool. Reply only with a short acknowledgement that another iteration is
  needed.
