from pathlib import Path
from typing import Final

CHAT_PROVIDERS: Final[tuple[str, ...]] = ("gemini", "openai", "anthropic")
IMAGE_PROVIDERS: Final[tuple[str, ...]] = ("gemini", "openai", "stability")

# "fast" runs generate -> evaluate in a plain Python for-loop and reads the
# stop/continue decision straight from evaluation state (no per-iteration LLM
# calls). "thinking" drives the same steps through ADK's LoopAgent and will
# grow richer reasoning over time. See pondercanvas.agent.refinement.
REFINEMENT_MODES: Final[tuple[str, ...]] = ("fast", "thinking")
DEFAULT_REFINEMENT_MODE: Final[str] = "fast"

DEFAULT_CHAT_PROVIDER: Final[str] = "gemini"
DEFAULT_CHAT_MODEL_ID: Final[str] = "gemini-3.5-flash"
DEFAULT_IMAGE_PROVIDER: Final[str] = "gemini"
DEFAULT_IMAGE_MODEL_ID: Final[str] = "gemini-3.1-flash-image"
DEFAULT_STRUCTURED_MODEL_ID: Final[str] = "gemini-3.5-flash"

MAX_ITERATIONS_CAP: Final[int] = 5
DEFAULT_MAX_ITERATIONS: Final[int] = 5
DEFAULT_EVAL_PASS_THRESHOLD: Final[float] = 4.0

DEFAULT_SIGLIP_ENABLED: Final[bool] = False
DEFAULT_SIGLIP_WEIGHT: Final[float] = 0.3
DEFAULT_SIGLIP_MODEL_ID: Final[str] = "google/siglip-base-patch16-224"
DEFAULT_ASPECT_RATIO: Final[str] = "1:1"
DEFAULT_OUTPUT_DIR: Final[Path] = Path("./.pondercanvas_runs")

DEFAULT_GEMINI_IMAGE_ENTERPRISE: Final[bool] = False

DEFAULT_MAX_REFERENCE_DOWNLOADS: Final[int] = 3
DEFAULT_MAX_DOWNLOAD_BYTES: Final[int] = 5_000_000
DEFAULT_DOWNLOAD_TIMEOUT_S: Final[float] = 5.0

UNSPLASH_UTM_SOURCE: Final[str] = "pondercanvas"
UNSPLASH_HOMEPAGE_URL: Final[str] = (
    f"https://unsplash.com/?utm_source={UNSPLASH_UTM_SOURCE}&utm_medium=referral"
)

SECRET_FIELD_SUFFIX: Final[str] = "_api_key"
REDACTED_MARKER: Final[str] = "***REDACTED***"
