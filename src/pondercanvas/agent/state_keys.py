"""session.state key constants. Every key here is written directly by a
tool/step function, never parsed out of an LLM's natural-language reply --
see the "state as truth, not LLM prose" principle in the design plan."""

BRIEF = "brief"
GROUNDING_RESULT = "grounding_result"
REFERENCE_IMAGE_BYTES = "reference_image_bytes"
LAST_IMAGE_PATH = "last_image_path"
LAST_EVALUATION = "last_evaluation"
ITERATIONS = "iterations"
