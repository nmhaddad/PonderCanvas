import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# W&B's own entity/project naming, not PONDERCANVAS_-prefixed like the rest of
# this app's settings: Weave reads WANDB_API_KEY/WANDB_PROJECT itself (via
# wandb's auth) rather than through our AppSettings.
_WANDB_PROJECT_DEFAULT = "nhaddad2112-duckasaurus/PonderCanvas"


def configure_tracing() -> None:
    """Initializes W&B Weave tracing for this process. Auto-patches the
    Gemini/OpenAI/Anthropic SDK calls the pipeline makes, plus the root spans
    added via @weave.op on the pipeline's entry points (agent/pipeline.py,
    agent/extraction.py, agent/refinement.py, providers/search/collect.py).

    No-ops (with a log message) when WANDB_API_KEY isn't configured, so local
    dev without W&B access still works. Safe to call more than once; idempotent
    like configure_logging."""
    load_dotenv()
    if not os.environ.get("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not set; Weave tracing disabled.")
        return

    import weave

    project = os.environ.get("WANDB_PROJECT", _WANDB_PROJECT_DEFAULT)
    weave.init(project)
    logger.info("Weave tracing initialized for project %s", project)
