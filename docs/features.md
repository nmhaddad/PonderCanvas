# Feature deep-dives

## Google image search (Gemini image generation)

Alongside the pre-loop Unsplash/Google-Search-text grounding described below, the Gemini image model can be given its own `google_search` tool (with `image_search` enabled) so it can pull in Google Image Search results as extra visual grounding at generation time — a second, independent source of reference imagery, not a replacement for Unsplash.

- **Toggle**: `PONDERCANVAS_GEMINI_IMAGE_SEARCH_ENABLED` (env) or the "Enable Google image search during generation" checkbox in the Settings panel. On by default; only applies to the Gemini image provider. No separate API key needed — it rides on the same Gemini image API key/client as generation itself.

## Tracking iterations via interaction ID, not re-uploaded bytes

Each `generate_image` call still writes the resulting image to disk (`<output_dir>/iteration_N.<ext>`) so `evaluate_image` and the UI have real bytes to score/display, and each iteration in the run trace records its own `interaction_id` for debugging. But *continuity* between iterations — the mechanism that lets the model edit its own prior output — no longer depends on that file: it's carried purely by `previous_interaction_id` in session state (`agent/state_keys.py::LAST_INTERACTION_ID`).

## Reference photos (Unsplash)

`collect_references()` always grounds the brief in Gemini's Google Search text results. Optionally, it also fetches real reference photos from [Unsplash](https://unsplash.com/developers) to pass alongside the user's own reference images:

- **Enable**: set `PONDERCANVAS_UNSPLASH_API_KEY` (env) or the "Unsplash Access Key" field in the Settings panel to an Unsplash Access Key. Without one, this step is skipped entirely (not an error) and only text grounding runs.
- **Behavior**: for each of the brief's search queries, up to `PONDERCANVAS_MAX_REFERENCE_DOWNLOADS` (default 3) safe-filtered (`content_filter=high`) photos are downloaded, each capped at `PONDERCANVAS_MAX_DOWNLOAD_BYTES` (default 5MB); any query or individual photo that fails is skipped rather than failing the run.
- **Attribution**: Unsplash's API guidelines require crediting the photographer and Unsplash for every photo actually used, with a tracked download ping per photo (`providers/search/unsplash_search.py`). This app does both automatically: "Photo by *Name* on Unsplash" (each linking to the photographer's profile and Unsplash, with `utm_source`/`utm_medium` as required) is rendered in the Gradio iteration trace for every run that used one.

## Evaluation scoring

By default, the `evaluate_image` tool scores each candidate purely on Gemini's structured critique (prompt adherence, aesthetic/technical quality, reference alignment). Optionally, a [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) image/text similarity score can be blended in as an additional, model-independent signal:

- **Toggle**: `PONDERCANVAS_SIGLIP_ENABLED` (env) or the "Enable SigLIP scoring" checkbox in the Settings panel. Off by default.
- **Weight**: `PONDERCANVAS_SIGLIP_WEIGHT` (env, 0.0–1.0) or the adjacent slider. SigLIP's score is rescaled from its native `[0, 1]` onto Gemini's `1–5` scale, then blended as `(1 - weight) * gemini_overall + weight * siglip_scaled` — the higher the weight, the more SigLIP's opinion counts relative to Gemini's, and the pass/fail decision is recomputed against `eval_pass_threshold` using this blended score.
- **Dependencies**: SigLIP scoring needs `torch` + `transformers`, an optional extra (not installed by default) since they're large: `uv sync --extra siglip`. If enabled but the model can't be loaded (extra not installed, no network to fetch weights, etc.), it logs a warning and falls back to Gemini's evaluation alone for that run rather than failing the pipeline.

## Tracing (W&B Weave)

Optional: set `WANDB_API_KEY` in `.env` (get one at [wandb.ai/authorize](https://wandb.ai/authorize)) to send a full trace of each run — extraction, grounding, every generate/evaluate iteration, and the underlying Gemini/OpenAI/Anthropic calls — to [Weights & Biases Weave](https://weave-docs.wandb.ai/). `WANDB_PROJECT` (`entity/project` format) controls where traces land; see `.env.example`. Leave `WANDB_API_KEY` blank to skip tracing entirely — nothing else changes.

### Known issue: `weave`/`google-adk` version mismatch breaks "thinking" mode tracing

`weave` 0.53.1 (latest published as of 2026-07-10) patches ADK's `trace_inference_result` with a wrapper written against an older ADK signature. `google-adk` 2.4.0 (required here for Workflow-graph support in `thinking` mode) calls it with an extra argument, so every "thinking"-mode run crashes with:

```
TypeError: trace_inference_result() takes 2 positional arguments but 3 were given
```

Worked around in `pondercanvas/tracing.py`: `google.adk` is pre-marked as "already patched" in Weave's internal registry before `weave.init()` runs, so Weave's implicit patcher skips just that one broken wrapper. ADK's own native tracing spans are unaffected, and this app's own `@weave.op` traces (pipeline, extraction, refinement, `collect_references`) are a separate mechanism and still fully work — the only loss is Weave's extra enrichment of ADK's model-call spans specifically. Remove the workaround once `weave` ships a `google-adk`-2.4-compatible integration.
