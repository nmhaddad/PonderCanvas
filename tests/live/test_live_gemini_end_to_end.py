import os

import pytest

from pondercanvas.agent.pipeline import PonderCanvasPipeline
from pondercanvas.config.settings import AppSettings, resolve_settings

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not os.environ.get("PONDERCANVAS_GOOGLE_API_KEY"),
    reason="requires a real PONDERCANVAS_GOOGLE_API_KEY",
)
@pytest.mark.asyncio
async def test_full_pipeline_run_against_real_gemini(tmp_path):
    """Real end-to-end smoke test: extraction, grounding, generation, and
    evaluation all hit live Gemini APIs. Only runs when explicitly opted
    into via PONDERCANVAS_RUN_LIVE_TESTS=1 and a real API key -- see
    tests/conftest.py and README.md."""
    settings = resolve_settings(AppSettings(output_dir=tmp_path, max_iterations=2))
    pipeline = PonderCanvasPipeline(settings)

    trace = await pipeline.run("a simple red apple on a white background", [])

    assert trace.final_image_path is not None
    assert len(trace.iterations) >= 1
    assert trace.stopped_reason in ("passed", "max_iterations_reached")
