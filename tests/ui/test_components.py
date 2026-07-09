from datetime import UTC, datetime

from pondercanvas.schemas.evaluation import EvaluationResult
from pondercanvas.schemas.grounding import GroundingResult, PhotoAttribution
from pondercanvas.schemas.trace import IterationTrace, RunTrace
from pondercanvas.ui.components import render_trace
from tests.fixtures.sample_brief import sample_brief


def _trace(**overrides) -> RunTrace:
    now = datetime.now(UTC)
    defaults: dict = dict(
        run_id="r1",
        brief=sample_brief(),
        iterations=[],
        passed=False,
        stopped_reason="max_iterations_reached",
        created_at=now,
    )
    defaults.update(overrides)
    return RunTrace(**defaults)


def _iteration(idx, passing, feedback="fb", overall=4.0) -> IterationTrace:
    return IterationTrace(
        iteration_index=idx,
        prompt_used="p",
        image_path=f"/tmp/{idx}.png",
        evaluation=EvaluationResult(
            scores={
                "prompt_adherence": overall,
                "aesthetic_quality": overall,
                "technical_quality": overall,
                "reference_alignment": overall,
            },
            overall=overall,
            is_passing=passing,
            feedback=feedback,
            threshold=4.0,
        ),
        created_at=datetime.now(UTC),
    )


class TestRenderTrace:
    def test_no_iterations_shows_placeholder_row(self):
        html_out = render_trace(_trace())
        assert "No iterations recorded" in html_out

    def test_includes_pass_and_fail_rows(self):
        trace = _trace(
            iterations=[_iteration(0, False), _iteration(1, True)], passed=True, stopped_reason="passed"
        )
        html_out = render_trace(trace)
        assert "FAIL" in html_out
        assert "PASS" in html_out

    def test_status_reflects_passed(self):
        trace = _trace(iterations=[_iteration(0, True)], passed=True, stopped_reason="passed")
        html_out = render_trace(trace)
        assert "Passed" in html_out

    def test_status_reflects_max_iterations(self):
        trace = _trace(iterations=[_iteration(0, False)], passed=False)
        html_out = render_trace(trace)
        assert "Reached max iterations" in html_out

    def test_feedback_is_html_escaped(self):
        trace = _trace(iterations=[_iteration(0, False, feedback="<script>alert(1)</script>")])
        html_out = render_trace(trace)
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out

    def test_iteration_index_displayed_as_one_based(self):
        trace = _trace(iterations=[_iteration(0, True)])
        html_out = render_trace(trace)
        assert "<td>1</td>" in html_out

    def test_no_grounding_shows_no_attribution(self):
        html_out = render_trace(_trace())
        assert ">Photo<" not in html_out

    def test_grounding_with_no_photos_shows_no_attribution(self):
        trace = _trace(grounding=GroundingResult(summary_text="ctx"))
        html_out = render_trace(trace)
        assert ">Photo<" not in html_out

    def test_photo_attribution_is_rendered_with_links(self):
        trace = _trace(
            grounding=GroundingResult(
                photo_attributions=[
                    PhotoAttribution(
                        photographer_name="Alice",
                        photographer_profile_url="https://unsplash.com/@alice",
                        photo_page_url="https://unsplash.com/photos/p1",
                    )
                ]
            )
        )
        html_out = render_trace(trace)
        assert ">Photo<" in html_out
        assert 'href="https://unsplash.com/photos/p1"' in html_out
        assert "by" in html_out
        assert 'href="https://unsplash.com/@alice"' in html_out
        assert ">Alice<" in html_out
        assert "unsplash.com/?utm_source=pondercanvas" in html_out
        assert ">Unsplash<" in html_out

    def test_multiple_photo_attributions_are_all_rendered(self):
        trace = _trace(
            grounding=GroundingResult(
                photo_attributions=[
                    PhotoAttribution(
                        photographer_name="Alice",
                        photographer_profile_url="https://u/a",
                        photo_page_url="https://u/photos/a",
                    ),
                    PhotoAttribution(
                        photographer_name="Bob",
                        photographer_profile_url="https://u/b",
                        photo_page_url="https://u/photos/b",
                    ),
                ]
            )
        )
        html_out = render_trace(trace)
        assert ">Alice<" in html_out
        assert ">Bob<" in html_out

    def test_photographer_name_is_html_escaped(self):
        trace = _trace(
            grounding=GroundingResult(
                photo_attributions=[
                    PhotoAttribution(
                        photographer_name="<script>alert(1)</script>",
                        photographer_profile_url="https://unsplash.com/@x",
                        photo_page_url="https://unsplash.com/photos/x",
                    )
                ]
            )
        )
        html_out = render_trace(trace)
        assert "<script>alert" not in html_out
        assert "&lt;script&gt;" in html_out
