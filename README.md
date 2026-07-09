<p align="center">
  <img src="media/pondercanvas.png" alt="PonderCanvas logo" width="200">
</p>

# PonderCanvas

PonderCanvas is an image-generation agent, inspired by Meta's [Muse](https://ai.meta.com/blog/introducing-muse-image-muse-video-msl/) blog post. You give it a text prompt and optional reference images; it extracts a structured brief, grounds itself with real web search, then iterates through a generate → evaluate → refine loop (up to `max_iterations` rounds, default 3) until the result passes a quality bar or the budget runs out.

## Architecture

```
User prompt + reference images
        │
        ▼
 extract_generation_brief()      ── plain call, always Gemini structured output
        │
        ▼
 collect_references()            ── plain call: Gemini Google Search grounding (text + citations)
        │                             + Unsplash reference photos (real image URLs, downloaded)
        ▼
 ┌─────────────────────────────────────────────┐
 │  Refinement loop (max_iterations, default 3) │
 │                                               │
 │   generate_image → evaluate_image → repeat    │
 │   loop exits early once evaluation passes     │
 │                                               │
 │   fast: plain Python for-loop (default)       │
 │   thinking: ADK Workflow graph of LlmAgents   │
 │     + a deterministic stop-check node,        │
 │     self-authored prompts + optional search   │
 │   instant: single generate_image call,        │
 │     no loop, no evaluation                    │
 └─────────────────────────────────────────────┘
        │
        ▼
   RunTrace (final image + per-iteration scores/feedback) → Gradio UI
```

Two design choices worth knowing about:

- **Extraction and evaluation always use Gemini**, independent of your chosen chat/image provider — they rely on Gemini's structured JSON output and (for grounding) Gemini's Google Search tool, neither of which has an equivalent in this project's other providers yet.
- **Refinement edits the previous image, it doesn't re-roll from scratch.** The first generation uses the user's/grounding reference images; every subsequent iteration feeds the model *its own previous output* as the only input image and reframes the critique as "corrections to apply to this image" (see `agent/tools/generation_tool.py` and `prompts/templates/generation_prompt.md.j2`). Without this, each pass regenerates a brand-new scene from the brief and the same structural flaw (e.g. a floating camera with no strap) recurs no matter how good the feedback text is — the model can only fix an artifact it can actually see.
- **The refinement loop has three selectable modes** (`PONDERCANVAS_REFINEMENT_MODE` or the "Refinement mode" dropdown in Settings; see `agent/refinement.py`):
  - **`fast`** (default) runs `generate_image → evaluate_image` in a plain Python `for` loop and stops the moment an evaluation passes. The stop/continue decision is already computed in Python (`evaluate_image`'s `pass` flag) and the evaluator's feedback reaches the next generation through session state, so this needs **zero LLM calls for orchestration**. The prompt fed to the image model is built from a fixed Jinja template (`prompts/templates/generation_prompt.md.j2`).
  - **`thinking`** drives the same two steps through a real `google.adk.workflow.Workflow` graph (see `agent/workflow.py`) — `GenerationAgent` and `EvaluationAgent` are each an ADK `LlmAgent` backed by your chosen chat model, wired in a cycle through a third graph node, `check_stop_condition` (see `agent/nodes.py`). That node is a plain deterministic function, not an LLM agent — it reads the same `pass` flag `fast` mode does and routes the graph either back to `GenerationAgent` or to a stop, so ending the loop costs no extra chat-model call. This replaces the deprecated `LoopAgent` primitive, which required a third LLM-backed agent (`LoopControlAgent`) just to decide when to call an `exit_loop` tool. Unlike `fast`, `GenerationAgent` composes the image prompt itself (see `prompts/templates/generation_instruction.md`) and can optionally call the tools `search_reference_images` (Unsplash) or `search_web` (Google Search) mid-loop when it decides current context doesn't cover something evaluation feedback called out — e.g. feedback wants a "wet" cat but existing references are all dry, so it searches "wet cat" (see `agent/tools/research.py`). It costs an extra chat-model round-trip per `LlmAgent` step per iteration, plus any research calls it chooses to make.
  - **`instant`** skips the loop and evaluation entirely: one `generate_image` call using the same fixed template as `fast`, for when you just want a single image out of the preloop extraction/grounding work with no refinement spend.

Both the **chat model** (drives the agent's tool-calling reasoning) and the **image-generation model** are swappable independently, via environment variables or live in the Settings panel:

| | Gemini | OpenAI | Anthropic | Stability |
|---|---|---|---|---|
| Chat model | native ADK `Gemini` | via LiteLLM | via LiteLLM | — |
| Image model | implemented | not yet implemented | — | not yet implemented |

Selecting `openai`/`stability` as the image provider fails loudly with `NotImplementedError` rather than silently falling back — the seam (`ImageProvider` in `providers/image/base.py`, registered in `providers/image/registry.py`) is ready for a real implementation once there's a key to test against.

### Gemini Enterprise / Vertex AI mode (image generation)

Some Gemini image models -- and some API-key restriction setups -- only work through the Gemini Enterprise Agent Platform (formerly Vertex AI) endpoint rather than the standard Gemini Developer API, and a single key's restrictions often can't be configured to allow both at once. If image generation fails with an access/permission error while chat/extraction work fine (which use the Developer API endpoint and aren't affected by this), that's the likely cause.

- **Toggle**: `PONDERCANVAS_GEMINI_IMAGE_ENTERPRISE` (env) or the "Use Gemini Enterprise/Vertex AI endpoint for image generation" checkbox in the Settings panel. This still authenticates with a plain API key ("Express Mode": `genai.Client(api_key=..., enterprise=True)`) -- no service account or `gcloud` ADC login required.
- **Key**: `PONDERCANVAS_GEMINI_IMAGE_API_KEY` (env) or the adjacent "Gemini image API key" field, if image generation needs a distinct key/restrictions from the one used for chat/extraction/grounding. Defaults to the main `PONDERCANVAS_GOOGLE_API_KEY` when left blank.
- Only affects the Gemini image provider (`providers/image/gemini_image.py`); ADK's own `Gemini` chat model has an equivalent knob if you ever need it there too (subclass `Gemini` and override `api_client` to return `Client(enterprise=True, ...)` -- see the ADK docs).
- **Image-only response modality**: `generate_content` is always called with `response_modalities=["IMAGE"]` (never `["TEXT", "IMAGE"]`). With TEXT allowed as a response modality, these models will sometimes return a prose *description* of the image (`finish_reason=STOP`, no error) instead of drawing it -- especially on elaborate prompts with reference images. Omitting TEXT removes that escape hatch. `output_mime_type` in `image_config` is the only knob gated to enterprise mode (the Developer API rejects it); `aspect_ratio` works on both.

### Reference photos (Unsplash)

`collect_references()` always grounds the brief in Gemini's Google Search text results. Optionally, it also fetches real reference photos from [Unsplash](https://unsplash.com/developers) to pass alongside the user's own reference images:

- **Enable**: set `PONDERCANVAS_UNSPLASH_API_KEY` (env) or the "Unsplash Access Key" field in the Settings panel to an Unsplash Access Key. Without one, this step is skipped entirely (not an error) and only text grounding runs.
- **Behavior**: for each of the brief's search queries, up to `PONDERCANVAS_MAX_REFERENCE_DOWNLOADS` (default 3) safe-filtered (`content_filter=high`) photos are downloaded, each capped at `PONDERCANVAS_MAX_DOWNLOAD_BYTES` (default 5MB); any query or individual photo that fails is skipped rather than failing the run.
- **Attribution**: Unsplash's API guidelines require crediting the photographer and Unsplash for every photo actually used, with a tracked download ping per photo (`providers/search/unsplash_search.py`). This app does both automatically: "Photo by *Name* on Unsplash" (each linking to the photographer's profile and Unsplash, with `utm_source`/`utm_medium` as required) is rendered in the Gradio iteration trace for every run that used one.

### Evaluation scoring

By default, the `evaluate_image` tool scores each candidate purely on Gemini's structured critique (prompt adherence, aesthetic/technical quality, reference alignment). Optionally, a [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) image/text similarity score can be blended in as an additional, model-independent signal:

- **Toggle**: `PONDERCANVAS_SIGLIP_ENABLED` (env) or the "Enable SigLIP scoring" checkbox in the Settings panel. Off by default.
- **Weight**: `PONDERCANVAS_SIGLIP_WEIGHT` (env, 0.0–1.0) or the adjacent slider. SigLIP's score is rescaled from its native `[0, 1]` onto Gemini's `1–5` scale, then blended as `(1 - weight) * gemini_overall + weight * siglip_scaled` — the higher the weight, the more SigLIP's opinion counts relative to Gemini's, and the pass/fail decision is recomputed against `eval_pass_threshold` using this blended score.
- **Dependencies**: SigLIP scoring needs `torch` + `transformers`, an optional extra (not installed by default) since they're large: `uv sync --extra siglip`. If enabled but the model can't be loaded (extra not installed, no network to fetch weights, etc.), it logs a warning and falls back to Gemini's evaluation alone for that run rather than failing the pipeline.

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Google API key (for Gemini structured extraction/evaluation and Google Search grounding) — required regardless of which chat/image provider you pick
- Optionally: an OpenAI key and/or an Anthropic key, depending on which providers you use
- Optionally: an Unsplash Access Key (unsplash.com/developers) if you want real reference photos alongside text grounding
- Optionally: `torch` + `transformers` (`uv sync --extra siglip`) if you want to enable SigLIP-based evaluation scoring

## Setup

```bash
uv sync                      # installs runtime + dev dependencies into .venv
cp .env.example .env         # fill in whatever keys you have; leave the rest blank
```

Every setting in `.env` can instead (or additionally) be set live in the Gradio Settings panel — nothing requires a restart. See `.env.example` for the full list of `PONDERCANVAS_*` variables.

## Running locally

```bash
uv run python -m pondercanvas
```

Opens a Gradio app at `http://localhost:7860`. Enter a prompt, optionally attach reference images, hit Generate, and watch the per-iteration trace (image, scores, feedback) fill in as the loop runs.

Logs (including full tracebacks for a failed Generate request) are written to `<output_dir>/pondercanvas.log` (`output_dir` defaults to `./.pondercanvas_runs`, see `PONDERCANVAS_OUTPUT_DIR`), rotated at 5MB with 3 backups kept, in addition to the console. Successful runs also get a JSONL summary at `<output_dir>/runs.jsonl`.

## Running tests

```bash
uv run pytest                                    # offline suite: no API keys, no network, ever
PONDERCANVAS_RUN_LIVE_TESTS=1 uv run pytest -m live   # end-to-end against real APIs (needs real keys)
```

The offline suite (`tests/unit/`, `tests/integration_offline/`, `tests/ui/`) mocks every external call — Gemini's client, LiteLLM, `requests` — including full `Workflow` graph executions driven by scripted fake models in `tests/fixtures/fake_llm.py`. Nothing in it needs credentials or a network connection. Live tests (`tests/live/`) are skipped by default and only run when explicitly opted into via the marker and env var above.

## Project layout

```
src/pondercanvas/
├── config/            settings.py (env + runtime overlay precedence), constants.py
├── schemas/           GenerationBrief, GroundingResult, EvaluationResult, RunTrace (Pydantic)
├── providers/
│   ├── chat/          build_chat_model(): Gemini natively, others via LiteLLM
│   ├── image/         ImageProvider interface + registry (Gemini implemented, others stubbed)
│   ├── structured/     GeminiStructuredVisionProvider (extraction + evaluation)
│   ├── scoring/         SiglipScorer: optional image/prompt similarity signal blended into evaluation
│   └── search/         Gemini Google Search grounding + Unsplash reference photos
├── agent/
│   ├── extraction.py   pre-loop: prompt/images -> GenerationBrief
│   ├── tools/           generate_image, evaluate_image, search_reference_images, search_web --
│   │                     ADK tool functions, only ever called *by* an LlmAgent (the latter two
│   │                     are thinking-mode only)
│   ├── nodes.py          check_stop_condition: a plain deterministic function wired directly
│   │                     into the thinking-mode graph's edges, never LLM-invoked -- not a tool
│   ├── agents.py         builds the GenerationAgent/EvaluationAgent LlmAgents
│   ├── workflow.py       build_refinement_workflow: wires the two LlmAgents + check_stop_condition
│   │                     into a google.adk.workflow.Workflow graph
│   ├── refinement.py    run_fast_refinement (for-loop) + run_thinking_refinement (Workflow graph)
│   │                     + run_instant_generation (single call, no loop)
│   └── pipeline.py      PonderCanvasPipeline: ties extraction -> grounding -> loop -> RunTrace
└── ui/                  Gradio app + settings panel + trace renderer

tests/
├── unit/                one file per provider/tool/schema/settings behavior, fully mocked
├── integration_offline/ real Workflow graph runs driven by scripted fake models, still offline
├── fixtures/             FakeImageProvider, FakeStructuredVisionProvider, FakeLlm variants
├── ui/                   Gradio callback logic tested as plain functions
└── live/                 real end-to-end run, gated behind the `live` marker
```

## Extending providers

- **New chat provider**: add one branch in `providers/chat/factory.py::build_chat_model` (any LiteLLM-supported provider is a one-line addition).
- **New image provider**: add a class implementing `ImageProvider` in `providers/image/`, register it in `providers/image/registry.py`.

## Known limitations

- Iterations are hard-capped at 5.
- Brief extraction, evaluation, and Google Search grounding always use Gemini, regardless of the chat/image provider you select.
- Local, single-process, single-user: settings live in server-side memory per browser session, not persisted to disk.
- OpenAI and Stability image providers are stubbed (raise `NotImplementedError`), pending API keys to implement and test against.
- Gemini image models occasionally return no image (`finish_reason=NO_IMAGE`) for a given prompt; this currently aborts the whole run rather than retrying. See the [issue tracker](https://github.com/nmhaddad/PonderCanvas/issues) for this and other open items.

## References

- [Introducing Muse: image & video generation](https://ai.meta.com/blog/introducing-muse-image-muse-video-msl) — Meta AI blog post that inspired this project's generate → evaluate → refine loop.
- [AI Agents for Image and Video Generation](https://www.deeplearning.ai/courses/ai-agents-for-image-and-video-generation) — DeepLearning.AI course that informed several implementation patterns here (structured extraction, image-generation call shape, critic/evaluator design).

## License

See [LICENSE](LICENSE).
