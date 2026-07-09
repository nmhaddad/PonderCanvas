import logging
from io import BytesIO
from typing import Any

from PIL import Image

from pondercanvas.config.constants import DEFAULT_SIGLIP_MODEL_ID

logger = logging.getLogger(__name__)


class SiglipScorer:
    """Scores image/prompt alignment with a SigLIP model as sigmoid(logits_per_image)
    in [0, 1]. Requires the optional 'siglip' dependency group (torch + transformers);
    if those aren't installed or the model fails to load, `score()` returns None
    instead of raising, after logging a warning once."""

    def __init__(self, model_id: str = DEFAULT_SIGLIP_MODEL_ID):
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None
        self._unavailable = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._processor is not None:
            return True
        if self._unavailable:
            return False
        try:
            from transformers import SiglipModel, SiglipProcessor

            self._model = SiglipModel.from_pretrained(self.model_id)
            self._processor = SiglipProcessor.from_pretrained(self.model_id)
        except Exception:
            logger.warning(
                "SigLIP scoring is enabled but model %r failed to initialize; "
                "disabling it for this run (install the optional dependencies with "
                "`uv sync --extra siglip` if they're missing).",
                self.model_id,
                exc_info=True,
            )
            # Reset in case the model loaded but the processor didn't (or vice
            # versa) -- otherwise the "already loaded" fast path above would
            # incorrectly report ready on the next call with a None processor.
            self._model = None
            self._processor = None
            self._unavailable = True
            return False
        return True

    def score(self, image_bytes: bytes, prompt: str) -> float | None:
        if not self._ensure_loaded():
            return None
        import torch

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(
            text=[prompt], images=[image], return_tensors="pt", padding="max_length"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        return torch.sigmoid(outputs.logits_per_image).item()
