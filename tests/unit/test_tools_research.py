import pondercanvas.agent.tools.research as research_module
from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.tools.research import make_search_reference_images_tool, make_search_web_tool
from pondercanvas.providers.search.unsplash_search import UnsplashPhoto
from pondercanvas.schemas.grounding import GroundingResult
from tests.fixtures.fake_tool_context import FakeToolContext


def _photo(id="p1", name="Alice") -> UnsplashPhoto:
    return UnsplashPhoto(
        id=id,
        image_url=f"https://img/{id}.jpg",
        photographer_name=name,
        photographer_profile_url=f"https://unsplash.com/@{name.lower()}",
        photo_page_url=f"https://unsplash.com/photos/{id}",
    )


class TestSearchReferenceImagesTool:
    def test_stores_downloaded_images_as_extra_reference_bytes(self, monkeypatch):
        monkeypatch.setattr(research_module, "search_photos", lambda *a, **k: [_photo()])
        monkeypatch.setattr(
            research_module, "download_photos", lambda *a, **k: [(b"wet-cat-bytes", _photo())]
        )
        tool = make_search_reference_images_tool("u-key", 3, 5_000_000, 5.0)
        ctx = FakeToolContext()

        result = tool("wet cat", ctx)

        assert ctx.state[sk.EXTRA_REFERENCE_IMAGE_BYTES] == [b"wet-cat-bytes"]
        assert result == {"status": "ok", "found": 1, "query": "wet cat"}

    def test_records_photo_attributions_and_accumulates_across_calls(self, monkeypatch):
        monkeypatch.setattr(research_module, "search_photos", lambda *a, **k: [_photo()])
        monkeypatch.setattr(
            research_module,
            "download_photos",
            lambda *a, **k: [(b"bytes", _photo(id="p1", name="Alice"))],
        )
        tool = make_search_reference_images_tool("u-key", 3, 5_000_000, 5.0)
        ctx = FakeToolContext()

        tool("wet cat", ctx)
        monkeypatch.setattr(
            research_module,
            "download_photos",
            lambda *a, **k: [(b"bytes2", _photo(id="p2", name="Bob"))],
        )
        tool("mossy rock", ctx)

        attributions = ctx.state[sk.PHOTO_ATTRIBUTIONS]
        assert [a["photographer_name"] for a in attributions] == ["Alice", "Bob"]

    def test_no_results_returns_empty_and_does_not_error(self, monkeypatch):
        monkeypatch.setattr(research_module, "search_photos", lambda *a, **k: [])
        monkeypatch.setattr(research_module, "download_photos", lambda *a, **k: [])
        tool = make_search_reference_images_tool("u-key", 3, 5_000_000, 5.0)
        ctx = FakeToolContext()

        result = tool("an extremely obscure query", ctx)

        assert result == {"status": "ok", "found": 0, "query": "an extremely obscure query"}
        assert ctx.state[sk.EXTRA_REFERENCE_IMAGE_BYTES] == []

    def test_passes_budget_settings_through_to_search_and_download(self, monkeypatch):
        search_calls = []
        download_calls = []
        monkeypatch.setattr(
            research_module,
            "search_photos",
            lambda query, api_key, max_results, timeout_s: search_calls.append(
                (query, api_key, max_results, timeout_s)
            )
            or [],
        )
        monkeypatch.setattr(
            research_module,
            "download_photos",
            lambda photos, api_key, max_images, max_bytes, timeout_s: download_calls.append(
                (api_key, max_images, max_bytes, timeout_s)
            )
            or [],
        )
        tool = make_search_reference_images_tool("u-key", 2, 999, 7.0)

        tool("wet cat", FakeToolContext())

        assert search_calls == [("wet cat", "u-key", 2, 7.0)]
        assert download_calls == [("u-key", 2, 999, 7.0)]


class TestSearchWebTool:
    def test_returns_summary_and_source_urls(self, monkeypatch):
        monkeypatch.setattr(
            research_module,
            "ground_with_search",
            lambda queries, api_key, model_id: GroundingResult(
                queries_used=queries,
                summary_text="wet cats have slicked-back fur",
                citations=[],
            ),
        )
        tool = make_search_web_tool("g-key", "gemini-3.5-flash")

        result = tool("wet cat fur")

        assert result["summary"] == "wet cats have slicked-back fur"

    def test_passes_query_api_key_and_model_id_through(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            research_module,
            "ground_with_search",
            lambda queries, api_key, model_id: calls.append((queries, api_key, model_id))
            or GroundingResult(queries_used=queries),
        )
        tool = make_search_web_tool("g-key", "gemini-3.5-flash")

        tool("wet cat fur")

        assert calls == [(["wet cat fur"], "g-key", "gemini-3.5-flash")]
