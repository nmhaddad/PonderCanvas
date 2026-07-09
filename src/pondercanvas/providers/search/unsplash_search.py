from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import requests

from pondercanvas.config.constants import UNSPLASH_UTM_SOURCE

_SEARCH_URL = "https://api.unsplash.com/search/photos"
_BASE_URL = "https://api.unsplash.com"


class UnsplashPhoto(NamedTuple):
    id: str
    image_url: str
    photographer_name: str
    photographer_profile_url: str
    photo_page_url: str


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Client-ID {api_key}"}


def search_photos(
    query: str, api_key: str, max_results: int = 5, timeout_s: float = 10.0
) -> list[UnsplashPhoto]:
    """Real photo search via the Unsplash API (unsplash.com/developers for an
    Access Key). Results are safe-filtered (content_filter=high)."""
    response = requests.get(
        _SEARCH_URL,
        headers=_auth_headers(api_key),
        params={"query": query, "per_page": str(min(max_results, 30)), "content_filter": "high"},
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()

    photos = []
    for item in data.get("results", []):
        image_url = item.get("urls", {}).get("regular")
        user = item.get("user") or {}
        username = user.get("username")
        photo_page_url = item.get("links", {}).get("html")
        if not image_url or not username or not photo_page_url:
            continue
        photos.append(
            UnsplashPhoto(
                id=item["id"],
                image_url=image_url,
                photographer_name=user.get("name") or username,
                photographer_profile_url=(
                    f"https://unsplash.com/@{username}"
                    f"?utm_source={UNSPLASH_UTM_SOURCE}&utm_medium=referral"
                ),
                photo_page_url=f"{photo_page_url}?utm_source={UNSPLASH_UTM_SOURCE}&utm_medium=referral",
            )
        )
    return photos


def _track_download(photo_id: str, api_key: str, timeout_s: float) -> None:
    """Pings Unsplash's download-tracking endpoint. Per the API guidelines,
    this must be called every time a photo's full-size bytes are actually
    retrieved for use (not just a hotlinked preview). Best-effort: a failure
    here shouldn't stop the photo we already fetched from being used."""
    try:
        requests.get(
            f"{_BASE_URL}/photos/{photo_id}/download",
            headers=_auth_headers(api_key),
            timeout=timeout_s,
        )
    except requests.RequestException:
        pass


def _download_one(
    photo: UnsplashPhoto, api_key: str, max_bytes: int, timeout_s: float
) -> tuple[bytes, UnsplashPhoto] | None:
    try:
        response = requests.get(photo.image_url, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException:
        return None
    if len(response.content) > max_bytes:
        return None
    _track_download(photo.id, api_key, timeout_s)
    return response.content, photo


def download_photos(
    photos: list[UnsplashPhoto],
    api_key: str,
    max_images: int,
    max_bytes: int,
    timeout_s: float,
) -> list[tuple[bytes, UnsplashPhoto]]:
    """Downloads up to max_images of the given photos, skipping any that
    error out or exceed max_bytes, and tracks each one actually used per
    Unsplash's API guidelines. Returns (image_bytes, photo) pairs so callers
    can attribute exactly the photos that ended up being used.

    Each photo's fetch (plus its tracking ping) runs on its own thread so the
    downloads happen concurrently rather than one blocking round-trip after
    another; input order is preserved in the returned list."""
    selected = photos[:max_images]
    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        results = executor.map(
            lambda photo: _download_one(photo, api_key, max_bytes, timeout_s), selected
        )
    return [result for result in results if result is not None]
