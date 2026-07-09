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
    def generate_image(tool_context: ToolContext) -> dict:
        """Generates a candidate image for the current brief via the configured
        image provider, incorporating grounding context and prior feedback, and
        records it as the latest iteration."""
        state = tool_context.state
        brief = state[sk.BRIEF]
        grounding = state.get(sk.GROUNDING_RESULT)
        feedback = state.get(sk.LAST_EVALUATION)
        prompt = build_generation_prompt(brief, grounding, feedback)

        reference_images = state.get(sk.REFERENCE_IMAGE_BYTES, [])
        result = image_provider.generate(prompt=prompt, reference_images=reference_images)

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
                "prompt_used": prompt,
                "image_path": str(image_path),
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        state[sk.ITERATIONS] = iterations

        return {"status": "ok", "image_path": str(image_path)}

    return generate_image
