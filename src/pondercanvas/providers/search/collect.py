from collections.abc import Callable

from pondercanvas.config.settings import EffectiveSettings
from pondercanvas.providers.search.gemini_search import ground_with_search
from pondercanvas.providers.search.unsplash_search import (
    UnsplashPhoto,
    download_photos,
    search_photos,
)
from pondercanvas.schemas.brief import GenerationBrief
from pondercanvas.schemas.grounding import GroundingResult, PhotoAttribution

GroundFn = Callable[[list[str], str | None, str], GroundingResult]
SearchPhotosFn = Callable[[str, str, int, float], list[UnsplashPhoto]]
DownloadPhotosFn = Callable[
    [list[UnsplashPhoto], str, int, int, float], list[tuple[bytes, UnsplashPhoto]]
]


def collect_references(
    brief: GenerationBrief,
    settings: EffectiveSettings,
    *,
    ground_fn: GroundFn = ground_with_search,
    search_photos_fn: SearchPhotosFn = search_photos,
    download_photos_fn: DownloadPhotosFn = download_photos,
) -> tuple[GroundingResult, list[bytes]]:
    """Runs once before the refinement loop (not per-iteration): grounds the
    brief in real Google Search text context, and -- if an Unsplash API key
    is configured -- fetches real reference photos. Text grounding always
    runs; photo references are skipped with an empty list (not an error)
    when no key is configured."""
    queries = brief.search_queries or [brief.subject]
    grounding = ground_fn(queries, settings.google_api_key, settings.structured_model_id)

    downloaded_images: list[bytes] = []
    attributions: list[PhotoAttribution] = []
    if settings.unsplash_api_key:
        photos: list[UnsplashPhoto] = []
        for query in queries:
            if len(photos) >= settings.max_reference_downloads:
                break
            photos.extend(
                search_photos_fn(
                    query,
                    settings.unsplash_api_key,
                    settings.max_reference_downloads,
                    settings.download_timeout_s,
                )
            )
        for image_bytes, photo in download_photos_fn(
            photos,
            settings.unsplash_api_key,
            settings.max_reference_downloads,
            settings.max_download_bytes,
            settings.download_timeout_s,
        ):
            downloaded_images.append(image_bytes)
            attributions.append(
                PhotoAttribution(
                    photographer_name=photo.photographer_name,
                    photographer_profile_url=photo.photographer_profile_url,
                    photo_page_url=photo.photo_page_url,
                )
            )

    grounding = grounding.model_copy(
        update={
            "downloaded_reference_count": len(downloaded_images),
            "photo_attributions": attributions,
        }
    )
    return grounding, downloaded_images
