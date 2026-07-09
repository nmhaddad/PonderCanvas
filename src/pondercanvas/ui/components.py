import base64
import html
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from pondercanvas.config.constants import UNSPLASH_HOMEPAGE_URL
from pondercanvas.schemas.trace import IterationTrace, RunTrace

_THUMBNAIL_MAX_SIZE = 96


def _thumbnail_data_uri(image_path: str) -> str | None:
    """Inline base64 thumbnail for the trace table: Gradio's HTML component
    can't serve arbitrary local paths as <img src>, so the resized image is
    embedded directly. Returns None (render no thumbnail) rather than
    raising if the file is missing or unreadable -- reading a path off disk
    at render time is a filesystem boundary, not something this function
    can guarantee about its input."""
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail((_THUMBNAIL_MAX_SIZE, _THUMBNAIL_MAX_SIZE))
            buffer = BytesIO()
            rgb_img.save(buffer, format="JPEG", quality=80)
    except (OSError, UnidentifiedImageError):
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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


def _render_iteration_row(it: IterationTrace, is_final: bool) -> str:
    thumbnail = _thumbnail_data_uri(it.image_path)
    thumbnail_cell = (
        f'<img src="{thumbnail}" width="{_THUMBNAIL_MAX_SIZE}" '
        f'alt="iteration {it.iteration_index + 1}">'
        if thumbnail
        else ""
    )
    ev = it.evaluation
    overall = f"{ev.overall:.2f}" if ev else "-"
    result = ("PASS" if ev.is_passing else "FAIL") if ev else "-"
    feedback = html.escape(ev.feedback) if ev else ""
    label = f"{it.iteration_index + 1}{' (final)' if is_final else ''}"
    return (
        f"<tr><td>{thumbnail_cell}</td><td>{label}</td><td>{result}</td>"
        f"<td>{overall}</td><td>{feedback}</td></tr>"
    )


def render_trace(trace: RunTrace) -> str:
    rows = [
        _render_iteration_row(it, is_final=it.image_path == trace.final_image_path)
        for it in trace.iterations
    ]

    body = "".join(rows) or "<tr><td colspan='5'>No iterations recorded</td></tr>"
    status = "Passed" if trace.passed else "Reached max iterations without passing"

    return (
        f"<p><strong>Status:</strong> {html.escape(status)} "
        f"({len(trace.iterations)} iteration(s))</p>"
        f"{_render_attribution(trace)}"
        "<table><thead><tr><th>Image</th><th>#</th><th>Result</th><th>Score</th>"
        "<th>Feedback</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )
