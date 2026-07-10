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

    # weave 0.53.1's google.adk integration wraps trace_inference_result with
    # a 2-arg (span, llm_response) signature, but google-adk>=2.4.0 (required
    # here for Workflow-graph support in "thinking" mode -- see
    # refinement.py) calls it with 3 args (invocation_context, span,
    # llm_response), crashing every "thinking"-mode run with "takes 2
    # positional arguments but 3 were given". No fixed weave release exists
    # yet (checked PyPI 2026-07-10) and downgrading google-adk would break
    # Workflow support instead. AutopatchSettings has no per-integration
    # toggle for google.adk, so pre-marking it "already patched" here is the
    # only way to skip just this one broken wrapper: weave.init()'s implicit
    # patcher (weave.integrations.patch.implicit_patch) only patches modules
    # not already in this set. ADK's own native OTel spans are unaffected --
    # this only forgoes Weave's extra enrichment of those specific spans, not
    # the @weave.op traces this app adds itself (pipeline.py/refinement.py/
    # collect.py), which are a separate, unaffected mechanism. Remove once
    # weave ships a compatible google-adk integration.
    from weave.integrations.patch import _PATCHED_INTEGRATIONS

    _PATCHED_INTEGRATIONS.add("google.adk")

    project = os.environ.get("WANDB_PROJECT", _WANDB_PROJECT_DEFAULT)
    weave.init(project)
    logger.info("Weave tracing initialized for project %s", project)
