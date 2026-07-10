from google import genai
from google.genai import types

from pondercanvas.providers._gemini import gemini_http_options
from pondercanvas.schemas.grounding import GroundingResult, SourceCitation


def ground_with_search(queries: list[str], model_id: str) -> GroundingResult:
    """Gemini's built-in Google Search grounding tool: returns grounded
    style/subject text plus citation URLs. Always Enterprise/Vertex AI +
    Application Default Credentials, no API key -- see AppSettings' comment
    near gemini_image_api_key."""
    client = genai.Client(enterprise=True, http_options=gemini_http_options())
    combined_query = "\n".join(queries) if queries else ""

    response = client.models.generate_content(
        model=model_id,
        contents=combined_query,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )

    citations: list[SourceCitation] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web is not None and getattr(web, "uri", None):
                citations.append(SourceCitation(url=web.uri, title=getattr(web, "title", None)))

    return GroundingResult(
        queries_used=queries,
        summary_text=getattr(response, "text", None) or "",
        citations=citations,
    )
