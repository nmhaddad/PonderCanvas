import html

from pondercanvas.config.constants import UNSPLASH_HOMEPAGE_URL
from pondercanvas.schemas.trace import RunTrace


def _render_attribution(trace: RunTrace) -> str:
    """Credit for any downloaded Unsplash reference photos: 'Photo by
    {photographer} on Unsplash', where 'Photo' links to the photo page
    (extra, not required) and {photographer}/'Unsplash' link to the
    profile/homepage (the credit Unsplash's API guidelines require). Empty
    string when no Unsplash photos were used."""
    attributions = trace.grounding.photo_attributions if trace.grounding else []
    if not attributions:
        return ""

    unsplash_link = (
        f'<a href="{html.escape(UNSPLASH_HOMEPAGE_URL)}" target="_blank" rel="noopener">Unsplash</a>'
    )
    credits = ", ".join(
        f'<a href="{html.escape(a.photo_page_url)}" target="_blank" rel="noopener">Photo</a> by '
        f'<a href="{html.escape(a.photographer_profile_url)}" '
        f'target="_blank" rel="noopener">{html.escape(a.photographer_name)}</a>'
        for a in attributions
    )
    return f"<p><small>{credits} on {unsplash_link}.</small></p>"


def render_trace(trace: RunTrace) -> str:
    rows = []
    for it in trace.iterations:
        ev = it.evaluation
        overall = f"{ev.overall:.2f}" if ev else "-"
        result = ("PASS" if ev.is_passing else "FAIL") if ev else "-"
        feedback = html.escape(ev.feedback) if ev else ""
        rows.append(
            f"<tr><td>{it.iteration_index + 1}</td><td>{result}</td>"
            f"<td>{overall}</td><td>{feedback}</td></tr>"
        )

    body = "".join(rows) or "<tr><td colspan='4'>No iterations recorded</td></tr>"
    status = "Passed" if trace.passed else "Reached max iterations without passing"

    return (
        f"<p><strong>Status:</strong> {html.escape(status)} "
        f"({len(trace.iterations)} iteration(s))</p>"
        f"{_render_attribution(trace)}"
        "<table><thead><tr><th>#</th><th>Result</th><th>Score</th><th>Feedback</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )
