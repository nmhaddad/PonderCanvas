import logging
from pathlib import Path

import gradio as gr

from pondercanvas.agent.pipeline import PonderCanvasPipeline
from pondercanvas.config.settings import AppSettings, resolve_settings
from pondercanvas.logging_utils import configure_logging
from pondercanvas.tracing import configure_tracing
from pondercanvas.ui.components import render_trace
from pondercanvas.ui.settings_panel import build_settings_panel, fields_to_overlay

logger = logging.getLogger(__name__)

_MISSING_IMAGE_KEY_MESSAGE = (
    "<p><strong>A Gemini image API key is required</strong> (Settings panel or "
    "PONDERCANVAS_GEMINI_IMAGE_API_KEY) for the Gemini image provider. Chat, extraction, "
    "evaluation, and search grounding always authenticate via Application Default "
    "Credentials instead, but image generation can't -- the Interactions API it uses "
    "isn't available for any model on that endpoint yet -- so an API key is still "
    "needed here.</p>"
)


async def _on_generate(prompt: str, files: list[str] | None, *settings_field_values):
    try:
        app_settings = AppSettings()
        overlay = fields_to_overlay(*settings_field_values)
        effective = resolve_settings(app_settings, overlay)

        if effective.image_provider == "gemini" and not effective.gemini_image_api_key:
            return None, _MISSING_IMAGE_KEY_MESSAGE

        reference_images = [Path(path).read_bytes() for path in (files or [])]
        pipeline = PonderCanvasPipeline(effective)
        trace = await pipeline.run(prompt, reference_images)
        return trace.final_image_path, render_trace(trace)
    except Exception:
        logger.exception("Generate failed for prompt=%r", prompt)
        raise


_LOGO_PATH = Path(__file__).resolve().parents[3] / "media" / "pondercanvas.png"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PonderCanvas") as demo:
        with gr.Row():
            gr.Image(
                value=str(_LOGO_PATH),
                show_label=False,
                container=False,
                interactive=False,
                height=80,
                width=80,
                scale=0,
                buttons=[],
            )
            gr.Markdown(
                "# PonderCanvas\n"
                "Text + optional reference images → grounded, self-critiquing image generation."
            )
        settings_fields = build_settings_panel()

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt", lines=3, placeholder="Describe the image you want..."
                )
                reference_images = gr.File(
                    label="Reference images (optional)",
                    file_count="multiple",
                    file_types=["image"],
                )
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Result")
                trace_html = gr.HTML(label="Iteration trace")

        generate_btn.click(
            _on_generate,
            inputs=[prompt, reference_images, *settings_fields],
            outputs=[output_image, trace_html],
        )
    return demo


def main() -> None:
    configure_logging(AppSettings().output_dir)
    configure_tracing()
    build_ui().launch()


if __name__ == "__main__":
    main()
