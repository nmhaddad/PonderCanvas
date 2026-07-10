from collections.abc import Callable

from google.adk.tools import ToolContext

from pondercanvas.agent import state_keys as sk
from pondercanvas.providers.search.gemini_search import ground_with_search
from pondercanvas.providers.search.unsplash_search import download_photos, search_photos
from pondercanvas.schemas.grounding import PhotoAttribution

SearchReferenceImagesTool = Callable[..., dict]
SearchWebTool = Callable[..., dict]


def make_search_reference_images_tool(
    api_key: str,
    max_results: int,
    max_bytes: int,
    timeout_s: float,
) -> SearchReferenceImagesTool:
    def search_reference_images(query: str, tool_context: ToolContext) -> dict:
        """Searches Unsplash for real reference photos matching `query` and
        makes them available to the very next generate_image call only -- they
        are not kept for later iterations. Call this only when there is a
        concrete visual gap current references and grounding don't cover, e.g.
        evaluation feedback wants a "wet" cat but existing references are all
        dry, so search "wet cat". Use a short, specific query, not a sentence,
        and skip this tool entirely when current context is already sufficient
        -- do not call it by default or "just in case"."""
        photos = search_photos(query, api_key, max_results, timeout_s)
        downloaded = download_photos(photos, api_key, max_results, max_bytes, timeout_s)

        state = tool_context.state
        state[sk.EXTRA_REFERENCE_IMAGE_BYTES] = [image_bytes for image_bytes, _ in downloaded]
        # Attribution accumulates for the whole run (not scoped to this turn
        # like the images themselves): any photo actually used in a generated
        # image must be credited in the final trace per Unsplash's API terms,
        # regardless of which iteration fetched it.
        new_attributions = [
            PhotoAttribution(
                photographer_name=photo.photographer_name,
                photographer_profile_url=photo.photographer_profile_url,
                photo_page_url=photo.photo_page_url,
            ).model_dump()
            for _, photo in downloaded
        ]
        state[sk.PHOTO_ATTRIBUTIONS] = [*state.get(sk.PHOTO_ATTRIBUTIONS, []), *new_attributions]

        return {"status": "ok", "found": len(downloaded), "query": query}

    return search_reference_images


def make_search_web_tool(model_id: str) -> SearchWebTool:
    def search_web(query: str) -> dict:
        """Searches Google for grounded text context about `query` and returns
        a short summary plus source URLs to weave into the next generation
        prompt. Call this only when the brief or current feedback references
        something needing factual or visual grounding current context doesn't
        already cover, and skip it otherwise -- do not call it by default or
        "just in case"."""
        result = ground_with_search([query], model_id)
        return {"summary": result.summary_text, "sources": [c.url for c in result.citations]}

    return search_web
