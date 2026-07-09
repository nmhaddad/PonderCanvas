import requests
import responses

from pondercanvas.providers.search import unsplash_search
from pondercanvas.providers.search.unsplash_search import (
    _BASE_URL,
    _SEARCH_URL,
    UnsplashPhoto,
    download_photos,
    search_photos,
)


def _result_item(
    id="p1",
    username="alice",
    name="Alice Photographer",
    url="https://img/regular.jpg",
    page_url="https://unsplash.com/photos/p1",
):
    return {
        "id": id,
        "urls": {"regular": url},
        "user": {"username": username, "name": name},
        "links": {"html": page_url},
    }


class TestSearchPhotos:
    @responses.activate
    def test_returns_photos_with_photographer_info(self):
        responses.add(
            responses.GET,
            _SEARCH_URL,
            json={"results": [_result_item()]},
            status=200,
        )

        photos = search_photos("bicycle", "api-key")

        assert photos == [
            UnsplashPhoto(
                id="p1",
                image_url="https://img/regular.jpg",
                photographer_name="Alice Photographer",
                photographer_profile_url=(
                    "https://unsplash.com/@alice?utm_source=pondercanvas&utm_medium=referral"
                ),
                photo_page_url=(
                    "https://unsplash.com/photos/p1?utm_source=pondercanvas&utm_medium=referral"
                ),
            )
        ]

    @responses.activate
    def test_falls_back_to_username_when_name_missing(self):
        responses.add(
            responses.GET,
            _SEARCH_URL,
            json={"results": [_result_item(name=None)]},
            status=200,
        )

        photos = search_photos("bicycle", "api-key")

        assert photos[0].photographer_name == "alice"

    @responses.activate
    def test_skips_items_missing_url_username_or_page_link(self):
        responses.add(
            responses.GET,
            _SEARCH_URL,
            json={
                "results": [
                    {"id": "no-url", "urls": {}, "user": {"username": "bob"}, "links": {"html": "x"}},
                    {
                        "id": "no-user",
                        "urls": {"regular": "https://img/x.jpg"},
                        "user": {},
                        "links": {"html": "x"},
                    },
                    {
                        "id": "no-page-link",
                        "urls": {"regular": "https://img/x.jpg"},
                        "user": {"username": "bob"},
                        "links": {},
                    },
                    _result_item(id="ok"),
                ]
            },
            status=200,
        )

        photos = search_photos("q", "api-key")

        assert [p.id for p in photos] == ["ok"]

    @responses.activate
    def test_no_results_returns_empty_list(self):
        responses.add(responses.GET, _SEARCH_URL, json={}, status=200)

        assert search_photos("q", "api-key") == []

    @responses.activate
    def test_sends_auth_header_and_content_filter(self):
        responses.add(responses.GET, _SEARCH_URL, json={"results": []}, status=200)

        search_photos("bicycle", "my-key")

        request = responses.calls[0].request
        assert request.headers["Authorization"] == "Client-ID my-key"
        assert "content_filter=high" in request.url

    @responses.activate
    def test_per_page_capped_at_thirty(self):
        responses.add(responses.GET, _SEARCH_URL, json={"results": []}, status=200)

        search_photos("q", "api-key", max_results=100)

        assert "per_page=30" in responses.calls[0].request.url


_PHOTO = UnsplashPhoto(
    id="p1",
    image_url="https://img/regular.jpg",
    photographer_name="Alice",
    photographer_profile_url="https://unsplash.com/@alice",
    photo_page_url="https://unsplash.com/photos/p1",
)


class TestDownloadPhotos:
    @responses.activate
    def test_downloads_bytes_and_returns_photo_pairs(self):
        responses.add(responses.GET, _PHOTO.image_url, body=b"image-bytes", status=200)
        responses.add(responses.GET, f"{_BASE_URL}/photos/p1/download", json={}, status=200)

        result = download_photos([_PHOTO], "api-key", max_images=5, max_bytes=1_000_000, timeout_s=5.0)

        assert result == [(b"image-bytes", _PHOTO)]

    @responses.activate
    def test_pings_download_tracking_endpoint_with_auth(self):
        responses.add(responses.GET, _PHOTO.image_url, body=b"data", status=200)
        responses.add(responses.GET, f"{_BASE_URL}/photos/p1/download", json={}, status=200)

        download_photos([_PHOTO], "my-key", max_images=5, max_bytes=1_000_000, timeout_s=5.0)

        tracking_call = responses.calls[1].request
        assert tracking_call.url == f"{_BASE_URL}/photos/p1/download"
        assert tracking_call.headers["Authorization"] == "Client-ID my-key"

    @responses.activate
    def test_skips_photo_exceeding_max_bytes_and_does_not_track_it(self):
        responses.add(responses.GET, _PHOTO.image_url, body=b"0123456789", status=200)

        result = download_photos([_PHOTO], "api-key", max_images=5, max_bytes=5, timeout_s=5.0)

        assert result == []
        assert len(responses.calls) == 1  # only the image fetch, no tracking ping

    @responses.activate
    def test_skips_photo_on_http_error_without_raising(self):
        responses.add(responses.GET, _PHOTO.image_url, status=500)

        result = download_photos([_PHOTO], "api-key", max_images=5, max_bytes=1_000_000, timeout_s=5.0)

        assert result == []

    @responses.activate
    def test_respects_max_images(self):
        photo_a = _PHOTO
        photo_b = UnsplashPhoto(
            "p2", "https://img/b.jpg", "Bob", "https://unsplash.com/@bob", "https://unsplash.com/photos/p2"
        )
        responses.add(responses.GET, photo_a.image_url, body=b"a", status=200)
        responses.add(responses.GET, f"{_BASE_URL}/photos/p1/download", json={}, status=200)

        result = download_photos(
            [photo_a, photo_b], "api-key", max_images=1, max_bytes=1_000_000, timeout_s=5.0
        )

        assert [photo for _, photo in result] == [photo_a]

    @responses.activate
    def test_tracking_endpoint_failure_does_not_drop_the_photo(self):
        responses.add(responses.GET, _PHOTO.image_url, body=b"data", status=200)
        responses.add(responses.GET, f"{_BASE_URL}/photos/p1/download", status=500)

        result = download_photos([_PHOTO], "api-key", max_images=5, max_bytes=1_000_000, timeout_s=5.0)

        assert result == [(b"data", _PHOTO)]


class _FakeResponse:
    def __init__(self, content: bytes = b""):
        self.content = content

    def raise_for_status(self) -> None:
        pass


def _photos(n: int) -> list[UnsplashPhoto]:
    return [
        UnsplashPhoto(f"p{i}", f"https://img/p{i}.jpg", f"N{i}", f"prof{i}", f"page{i}")
        for i in range(n)
    ]


class TestDownloadPhotosConcurrency:
    def test_preserves_input_order_regardless_of_completion_order(self, monkeypatch):
        # Downloads run on separate threads now; executor.map must still return
        # results in the input photo order, not whichever thread finished first.
        def fake_get(url, timeout=None, headers=None):
            if url.endswith("/download"):
                return _FakeResponse()
            photo_id = url.rsplit("/", 1)[-1].removesuffix(".jpg")
            return _FakeResponse(content=photo_id.encode())

        monkeypatch.setattr(unsplash_search.requests, "get", fake_get)

        result = download_photos(_photos(3), "k", max_images=3, max_bytes=1_000_000, timeout_s=1.0)

        assert [photo.id for _, photo in result] == ["p0", "p1", "p2"]
        assert [image_bytes for image_bytes, _ in result] == [b"p0", b"p1", b"p2"]

    def test_failed_download_is_filtered_out_while_order_is_preserved(self, monkeypatch):
        def fake_get(url, timeout=None, headers=None):
            if url.endswith("/download"):
                return _FakeResponse()
            photo_id = url.rsplit("/", 1)[-1].removesuffix(".jpg")
            if photo_id == "p1":
                raise requests.RequestException("boom")
            return _FakeResponse(content=photo_id.encode())

        monkeypatch.setattr(unsplash_search.requests, "get", fake_get)

        result = download_photos(_photos(3), "k", max_images=3, max_bytes=1_000_000, timeout_s=1.0)

        assert [photo.id for _, photo in result] == ["p0", "p2"]
