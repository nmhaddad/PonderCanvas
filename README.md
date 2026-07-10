<p align="center">
  <img src="media/pondercanvas.png" alt="PonderCanvas logo" width="200">
</p>

# PonderCanvas

PonderCanvas is an image-generation agent, inspired by Meta's [Muse](https://ai.meta.com/blog/introducing-muse-image-muse-video-msl/) blog post. You give it a text prompt and optional reference images; it extracts a structured brief, grounds itself with real web search, then iterates through a generate → evaluate → refine loop (up to `max_iterations` rounds, default 3) until the result passes a quality bar or the budget runs out.

Example output from the same prompt under each refinement mode:

<table>
  <tr>
    <td align="center"><img src="media/instant.jpeg" width="250"><br><sub><b>instant</b></sub></td>
    <td align="center"><img src="media/fast.jpeg" width="250"><br><sub><b>fast</b></sub></td>
    <td align="center"><img src="media/thinking.jpeg" width="250"><br><sub><b>thinking</b></sub></td>
  </tr>
</table>

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

Refinement edits the previous image rather than re-rolling from scratch — every iteration after the first continues the model's own prior generation via the Gemini Interactions API's `previous_interaction_id`, so a fix to one flaw doesn't undo everything else. Extraction and evaluation always use Gemini regardless of your chosen chat/image provider.

See **[docs/architecture.md](docs/architecture.md)** for the full breakdown: the three refinement modes (`fast`/`thinking`/`instant`), how chat/image providers are swapped and extended, and the project layout.

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — refinement modes, provider abstraction, project layout, extending with a new chat/image provider.
- **[docs/google-auth.md](docs/google-auth.md)** — how Gemini authenticates across this app (Application Default Credentials for chat/extraction/evaluation/search vs. an API key for image generation), plus troubleshooting for the permission errors each path can produce.
- **[docs/features.md](docs/features.md)** — Google image search, interaction-ID-based iteration tracking, Unsplash reference photos, SigLIP evaluation scoring, W&B Weave tracing (and a known `weave`/`google-adk` compatibility workaround).

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Working Application Default Credentials (`gcloud auth application-default login`, or `GOOGLE_APPLICATION_CREDENTIALS`) plus `GOOGLE_CLOUD_PROJECT` set — used for chat, structured extraction/evaluation, and Google Search grounding regardless of which chat/image provider you pick; see [docs/google-auth.md](docs/google-auth.md)
- A Gemini image API key (`PONDERCANVAS_GEMINI_IMAGE_API_KEY`) if you're using the Gemini image provider (the default) — ADC doesn't cover image generation
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

## Known limitations

- Iterations are hard-capped at 5.
- Brief extraction, evaluation, and Google Search grounding always use Gemini, regardless of the chat/image provider you select.
- Local, single-process, single-user: settings live in server-side memory per browser session, not persisted to disk.
- OpenAI and Stability image providers are stubbed (raise `NotImplementedError`), pending API keys to implement and test against.
- Gemini image models occasionally return no output image for a given prompt (safety block, refusal, empty response, etc.); this currently aborts the whole run rather than retrying. See the [issue tracker](https://github.com/nmhaddad/PonderCanvas/issues) for this and other open items.
- Gemini image generation does not work with `PONDERCANVAS_GEMINI_IMAGE_ENTERPRISE=true`: the Interactions API isn't onboarded on the Enterprise/Vertex AI endpoint yet, only the standard Developer API endpoint. See [docs/google-auth.md](docs/google-auth.md).

## References

- [Introducing Muse: image & video generation](https://ai.meta.com/blog/introducing-muse-image-muse-video-msl) — Meta AI blog post that inspired this project's generate → evaluate → refine loop.
- [AI Agents for Image and Video Generation](https://www.deeplearning.ai/courses/ai-agents-for-image-and-video-generation) — DeepLearning.AI course that informed several implementation patterns here (structured extraction, image-generation call shape, critic/evaluator design).

## License

See [LICENSE](LICENSE).
