"""session.state key constants. Every key here is written directly by a
tool/step function, never parsed out of an LLM's natural-language reply --
see the "state as truth, not LLM prose" principle in the design plan."""

BRIEF = "brief"
GROUNDING_RESULT = "grounding_result"
REFERENCE_IMAGE_BYTES = "reference_image_bytes"
LAST_IMAGE_PATH = "last_image_path"
LAST_EVALUATION = "last_evaluation"
ITERATIONS = "iterations"

# Thinking-mode-only: images the search_reference_images tool fetched this
# turn, scoped to the very next generate_image call only (cleared once
# consumed -- see tools/generation_tool.py).
EXTRA_REFERENCE_IMAGE_BYTES = "extra_reference_image_bytes"
# Accumulates across the whole run so every photo actually used in a
# generated image gets credited, even ones fetched mid-loop by
# search_reference_images (see tools/research.py and pipeline._assemble_run_trace).
PHOTO_ATTRIBUTIONS = "photo_attributions"
