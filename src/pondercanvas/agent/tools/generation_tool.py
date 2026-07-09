from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from google.adk.tools import ToolContext

from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.prompts import build_generation_prompt
from pondercanvas.providers.image.base import ImageProvider

GenerateImageTool = Callable[[ToolContext], dict]


def make_generate_image_tool(
    image_provider: ImageProvider, output_dir: Path
) -> GenerateImageTool:
    def generate_image(tool_context: ToolContext, prompt: str | None = None) -> dict:
        """Generates a candidate image for the current brief via the configured
        image provider, incorporating grounding context and prior feedback, and
        records it as the latest iteration. Pass `prompt`: one cohesive, vivid
        scene description weaving together the brief, grounding, any feedback
        corrections, and anything found via search_reference_images/search_web
        this turn -- concrete materials/textures and precise photographic
        framing terms, not a checklist or generic language. When omitted, a
        default prompt is built from the brief automatically."""
        state = tool_context.state
        brief = state[sk.BRIEF]
        grounding = state.get(sk.GROUNDING_RESULT)
        feedback = state.get(sk.LAST_EVALUATION)

        # On refinement iterations, feed the model its own previous image so it
        # *edits* that image to address the critique, rather than regenerating a
        # fresh scene from the brief and re-rolling the same flaw. Critique text
        # alone can't fix a specific rendered artifact the model never sees --
        # e.g. "add a strap to the floating camera" only lands if the model is
        # looking at the camera it drew.
        previous_image_path = state.get(sk.LAST_IMAGE_PATH)
        revising = bool(feedback and feedback.get("feedback") and previous_image_path)
        final_prompt = prompt or build_generation_prompt(brief, grounding, feedback, revising=revising)

        # Photos the search_reference_images tool fetched this turn (thinking
        # mode only) are scoped to this one generate call -- clear them here so
        # a later iteration doesn't silently keep reusing a stale search.
        extra_images = list(state.get(sk.EXTRA_REFERENCE_IMAGE_BYTES, []))
        if extra_images:
            state[sk.EXTRA_REFERENCE_IMAGE_BYTES] = []

        if revising:
            # After the first pass, refine the model's own previous output
            # directly: feed back ONLY that image (plus anything freshly
            # searched for), not the user's/grounding references. The first
            # render already absorbed those references; re-supplying them here
            # pulls the edit back toward them instead of fixing the specific
            # artifact in the last result. So the user's reference images are
            # used for the first generation only.
            generation_images = [Path(previous_image_path).read_bytes(), *extra_images]
        else:
            generation_images = [*state.get(sk.REFERENCE_IMAGE_BYTES, []), *extra_images]

        result = image_provider.generate(prompt=final_prompt, reference_images=generation_images)

        iterations = list(state.get(sk.ITERATIONS, []))
        idx = len(iterations)

        output_dir.mkdir(parents=True, exist_ok=True)
        extension = result.mime_type.split("/")[-1] or "png"
        image_path = output_dir / f"iteration_{idx}.{extension}"
        image_path.write_bytes(result.image_bytes)

        state[sk.LAST_IMAGE_PATH] = str(image_path)
        iterations.append(
            {
                "iteration_index": idx,
                "prompt_used": final_prompt,
                "image_path": str(image_path),
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        state[sk.ITERATIONS] = iterations

        return {"status": "ok", "image_path": str(image_path)}

    return generate_image
