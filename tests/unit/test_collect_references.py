from pondercanvas.config.settings import AppSettings, resolve_settings
from pondercanvas.providers.search.collect import collect_references
from pondercanvas.providers.search.unsplash_search import UnsplashPhoto
from pondercanvas.schemas.brief import GenerationBrief
from pondercanvas.schemas.grounding import GroundingResult


def _brief(**overrides) -> GenerationBrief:
    defaults = dict(
        subject="a red bicycle",
        style="watercolor",
        composition="centered",
        mood="cheerful",
        palette="warm",
        search_queries=["red bicycle watercolor"],
        raw_user_prompt="draw a red bicycle",
    )
    defaults.update(overrides)
    return GenerationBrief(**defaults)


def _effective(**overrides):
    return resolve_settings(AppSettings(_env_file=None, **overrides))  # type: ignore[call-arg]


def _photo(
    id="p1",
    name="Alice",
    profile="https://unsplash.com/@alice",
    page_url="https://unsplash.com/photos/p1",
) -> UnsplashPhoto:
    return UnsplashPhoto(
        id=id,
        image_url=f"https://img/{id}.jpg",
        photographer_name=name,
        photographer_profile_url=profile,
        photo_page_url=page_url,
    )


def _no_photos_search_fn(query, api_key, max_results, timeout_s):
    return []


def _no_photos_download_fn(photos, api_key, max_images, max_bytes, timeout_s):
    return []


class TestCollectReferencesGrounding:
    def test_runs_text_grounding_with_briefs_search_queries(self):
        ground_calls = []

        def fake_ground(queries, model_id):
            ground_calls.append(queries)
            return GroundingResult(queries_used=queries, summary_text="grounded")

        grounding, images = collect_references(
            _brief(), _effective(), ground_fn=fake_ground, search_photos_fn=_no_photos_search_fn
        )

        assert ground_calls == [["red bicycle watercolor"]]
        assert grounding.summary_text == "grounded"
        assert images == []

    def test_falls_back_to_subject_when_no_search_queries(self):
        ground_calls = []

        def fake_ground(queries, model_id):
            ground_calls.append(queries)
            return GroundingResult(queries_used=queries)

        collect_references(
            _brief(search_queries=[]),
            _effective(),
            ground_fn=fake_ground,
            search_photos_fn=_no_photos_search_fn,
        )

        assert ground_calls == [["a red bicycle"]]

    def test_passes_structured_model_id_to_ground_fn(self):
        calls = []

        def fake_ground(queries, model_id):
            calls.append(model_id)
            return GroundingResult(queries_used=queries)

        collect_references(
            _brief(),
            _effective(structured_model_id="gemini-custom"),
            ground_fn=fake_ground,
            search_photos_fn=_no_photos_search_fn,
        )

        assert calls == ["gemini-custom"]


class TestCollectReferencesUnsplash:
    def test_skips_photo_search_when_no_unsplash_key_configured(self):
        grounding, images = collect_references(
            _brief(),
            _effective(unsplash_api_key=None),
            ground_fn=lambda q, m: GroundingResult(queries_used=q),
            search_photos_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
            download_photos_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
        )
        assert images == []
        assert grounding.downloaded_reference_count == 0
        assert grounding.photo_attributions == []

    def test_searches_and_downloads_when_key_configured(self):
        search_calls = []
        photo = _photo()

        def fake_search(query, api_key, max_results, timeout_s):
            search_calls.append((query, api_key, max_results))
            return [photo]

        def fake_download(photos, api_key, max_images, max_bytes, timeout_s):
            return [(b"img-bytes", p) for p in photos]

        grounding, images = collect_references(
            _brief(),
            _effective(unsplash_api_key="u-key", max_reference_downloads=2),
            ground_fn=lambda q, m: GroundingResult(queries_used=q),
            search_photos_fn=fake_search,
            download_photos_fn=fake_download,
        )

        assert images == [b"img-bytes"]
        assert search_calls == [("red bicycle watercolor", "u-key", 2)]
        assert grounding.downloaded_reference_count == 1

    def test_grounding_records_photo_attribution_for_downloaded_photos_only(self):
        photo_kept = _photo(
            id="kept",
            name="Alice",
            profile="https://unsplash.com/@alice",
            page_url="https://unsplash.com/photos/kept",
        )
        photo_dropped = _photo(id="dropped", name="Bob", profile="https://unsplash.com/@bob")

        def fake_download(photos, api_key, max_images, max_bytes, timeout_s):
            return [(b"bytes", photo_kept)]  # simulates photo_dropped failing to download

        grounding, images = collect_references(
            _brief(),
            _effective(unsplash_api_key="u-key"),
            ground_fn=lambda q, m: GroundingResult(queries_used=q),
            search_photos_fn=lambda *a, **k: [photo_kept, photo_dropped],
            download_photos_fn=fake_download,
        )

        assert len(grounding.photo_attributions) == 1
        assert grounding.photo_attributions[0].photographer_name == "Alice"
        assert grounding.photo_attributions[0].photographer_profile_url == "https://unsplash.com/@alice"
        assert grounding.photo_attributions[0].photo_page_url == "https://unsplash.com/photos/kept"

    def test_stops_searching_once_max_reference_downloads_reached(self):
        search_calls = []

        def fake_search(query, api_key, max_results, timeout_s):
            search_calls.append(query)
            return [_photo(id=query)]

        collect_references(
            _brief(search_queries=["q1", "q2", "q3"]),
            _effective(unsplash_api_key="u-key", max_reference_downloads=1),
            ground_fn=lambda q, m: GroundingResult(queries_used=q),
            search_photos_fn=fake_search,
            download_photos_fn=_no_photos_download_fn,
        )

        assert search_calls == ["q1"]

    def test_passes_download_limits_through_to_download_photos_fn(self):
        download_calls = []

        def fake_download(photos, api_key, max_images, max_bytes, timeout_s):
            download_calls.append((api_key, max_images, max_bytes, timeout_s))
            return []

        collect_references(
            _brief(),
            _effective(
                unsplash_api_key="u-key",
                max_reference_downloads=4,
                max_download_bytes=123,
                download_timeout_s=9.0,
            ),
            ground_fn=lambda q, m: GroundingResult(queries_used=q),
            search_photos_fn=lambda *a, **k: [_photo()],
            download_photos_fn=fake_download,
        )

        assert download_calls == [("u-key", 4, 123, 9.0)]
