# Architecture

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
- **Refinement edits the previous image, it doesn't re-roll from scratch.** The first generation uses the user's/grounding reference images; every subsequent iteration continues the model's own previous generation via the Gemini [Interactions API](https://ai.google.dev/gemini-api/docs/image-generation)'s `previous_interaction_id` and reframes the critique as "corrections to apply to this image" (see `agent/tools/generation_tool.py` and `prompts/templates/generation_prompt.md.j2`) — the previous image never needs to be re-read from disk or re-uploaded; Gemini keeps it server-side and the run only carries its interaction ID forward in session state. Without this continuation, each pass would regenerate a brand-new scene from the brief and the same structural flaw (e.g. a floating camera with no strap) would recur no matter how good the feedback text is — the model can only fix an artifact it can actually see.

## Refinement modes

The refinement loop has three selectable modes (`PONDERCANVAS_REFINEMENT_MODE` or the "Refinement mode" dropdown in Settings; see `agent/refinement.py`):

- **`fast`** (default) runs `generate_image → evaluate_image` in a plain Python `for` loop and stops the moment an evaluation passes. The stop/continue decision is already computed in Python (`evaluate_image`'s `pass` flag) and the evaluator's feedback reaches the next generation through session state, so this needs **zero LLM calls for orchestration**. The prompt fed to the image model is built from a fixed Jinja template (`prompts/templates/generation_prompt.md.j2`).
- **`thinking`** drives the same two steps through a real `google.adk.workflow.Workflow` graph (see `agent/workflow.py`) — `GenerationAgent` and `EvaluationAgent` are each an ADK `LlmAgent` backed by your chosen chat model, wired in a cycle through a third graph node, `check_stop_condition` (see `agent/nodes.py`). That node is a plain deterministic function, not an LLM agent — it reads the same `pass` flag `fast` mode does and routes the graph either back to `GenerationAgent` or to a stop, so ending the loop costs no extra chat-model call. This replaces the deprecated `LoopAgent` primitive, which required a third LLM-backed agent (`LoopControlAgent`) just to decide when to call an `exit_loop` tool. Unlike `fast`, `GenerationAgent` composes the image prompt itself (see `prompts/templates/generation_instruction.md`) and can optionally call the tools `search_reference_images` (Unsplash) or `search_web` (Google Search) mid-loop when it decides current context doesn't cover something evaluation feedback called out — e.g. feedback wants a "wet" cat but existing references are all dry, so it searches "wet cat" (see `agent/tools/research.py`). It costs an extra chat-model round-trip per `LlmAgent` step per iteration, plus any research calls it chooses to make.
- **`instant`** skips the loop and evaluation entirely: one `generate_image` call using the same fixed template as `fast`, for when you just want a single image out of the preloop extraction/grounding work with no refinement spend.

## Chat and image providers

Both the **chat model** (drives the agent's tool-calling reasoning) and the **image-generation model** are swappable independently, via environment variables (`PONDERCANVAS_CHAT_PROVIDER` / `PONDERCANVAS_IMAGE_PROVIDER`) — there's no picker for these in the Gradio UI, only model-ID textboxes; provider selection is env-only:

| | Gemini | OpenAI | Anthropic | Stability |
|---|---|---|---|---|
| Chat model | native ADK `Gemini` | via LiteLLM | via LiteLLM | — |
| Image model | implemented | not yet implemented | — | not yet implemented |

Selecting `openai`/`stability` as the image provider fails loudly with `NotImplementedError` rather than silently falling back — the seam (`ImageProvider` in `providers/image/base.py`, registered in `providers/image/registry.py`) is ready for a real implementation once there's a key to test against.

See [google-auth.md](google-auth.md) for how Gemini itself authenticates (ADC vs. API key) across these paths.

## Extending providers

- **New chat provider**: add one branch in `providers/chat/factory.py::build_chat_model` (any LiteLLM-supported provider is a one-line addition).
- **New image provider**: add a class implementing `ImageProvider` in `providers/image/`, register it in `providers/image/registry.py`.

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
