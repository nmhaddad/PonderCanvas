# Google authentication: ADC vs. API keys

Gemini is used in two functionally separate places in this app, and they authenticate completely differently. Mixing them up is the single most common source of confusing 403/permission errors, so this doc spells out both paths precisely — each claim below was verified live against the real APIs, not just inferred from docs.

## Chat, extraction, evaluation, search grounding — Enterprise/Vertex AI + ADC

Chat (`build_chat_model`), `GeminiStructuredVisionProvider` (extraction/evaluation), and `ground_with_search` always authenticate via Application Default Credentials (`gcloud auth application-default login`, or `GOOGLE_APPLICATION_CREDENTIALS`) against the Enterprise Agent Platform (formerly Vertex AI) endpoint — there is no API key setting for these, and no toggle to turn ADC off. Confirmed working live against `generate_content` on that endpoint.

You'll need `GOOGLE_CLOUD_PROJECT` set in your environment (read directly by the `google-genai` SDK, not `PONDERCANVAS_*`) since ADC alone can't always infer a project — this is especially true if your ADC credentials are an impersonated service account (`type: impersonated_service_account` in `~/.config/gcloud/application_default_credentials.json`), which carries no project of its own. `GOOGLE_CLOUD_LOCATION` is optional (defaults to `global`).

If ADC itself is misconfigured (e.g. missing IAM permission to impersonate a target service account), you'll see something like:

```
RefreshError: Unable to acquire impersonated credentials
403: Permission 'iam.serviceAccounts.getAccessToken' denied on resource
```

Fix by granting your user the **Service Account Token Creator** role on the target service account in Cloud Console → IAM & Admin → Service Accounts, or by re-running `gcloud auth application-default login` without `--impersonate-service-account` if you don't actually need impersonation.

**Does not cover image generation.** Confirmed live: the Interactions API `GeminiImageProvider` uses is reachable on the Enterprise endpoint now (no more 404 at the gateway) but rejects every model tried, image and chat alike, with `400 Unsupported model interaction` — nothing is actually onboarded there yet. So the Gemini image provider keeps its own separate API key and its own separate enterprise toggle regardless of the above; see the next section.

## Image generation — Gemini Enterprise / Vertex AI mode

> **Currently broken for image generation.** Same underlying endpoint as above, but the Interactions API `GeminiImageProvider` uses isn't usable there for any model right now (`400 Unsupported model interaction`, confirmed live) — not the gateway-level 404 this used to produce. Enabling this toggle for image generation fails fast with a `RuntimeError` explaining why. If your API key's restrictions require the Enterprise endpoint, Gemini image generation isn't usable through this app until Google onboards models to Interactions-via-Vertex; chat/extraction/evaluation already use ADC unconditionally, independently of this toggle.

Some Gemini image models — and some API-key restriction setups — only work through the Gemini Enterprise Agent Platform (formerly Vertex AI) endpoint rather than the standard Gemini Developer API, and a single key's restrictions often can't be configured to allow both at once.

- **Toggle**: `PONDERCANVAS_GEMINI_IMAGE_ENTERPRISE` (env) or the "Use Gemini Enterprise/Vertex AI endpoint for image generation" checkbox in the Settings panel. This still authenticates with a plain API key ("Express Mode": `genai.Client(api_key=..., enterprise=True)`) — no service account or `gcloud` ADC login required.
- **Key**: `PONDERCANVAS_GEMINI_IMAGE_API_KEY` (env) or the "Gemini image API key" field in the Settings panel — required, no fallback to another key.
- Only affects the Gemini image provider (`providers/image/gemini_image.py`); independent of the chat/extraction/evaluation/search-grounding ADC path above.
- **Image-only response modality**: `client.interactions.create` is always called with `response_modalities=["image"]` — lowercase; the API 400s on uppercase `"IMAGE"` (confirmed live) — and never includes `"text"`. With text allowed as a response modality, these models will sometimes return a prose *description* of the image instead of drawing it — especially on elaborate prompts with reference images. Omitting it removes that escape hatch. `mime_type` in `response_format` is only ever sent (as `"image/jpeg"`) when the configured output MIME type is explicitly jpeg — the SDK only supports requesting jpeg explicitly, so PNG output (the default) is left to the API's own default rather than requested.

## Troubleshooting: `API_KEY_SERVICE_BLOCKED` / `PERMISSION_DENIED` on `interactions.create`

If `PONDERCANVAS_GEMINI_IMAGE_API_KEY` fails with something like:

```
403 PERMISSION_DENIED: Requests to this API generativelanguage.googleapis.com method
google.learning.gemini.api.interactions.v1beta.InteractionsService.CreateInteractionHttp
are blocked. reason: API_KEY_SERVICE_BLOCKED
```

this is an account/project-level restriction on that specific key — confirmed reproducible via a raw `curl` call outside the SDK, ruling out any PonderCanvas or `google-genai` bug. It is **not** fixed by switching to Enterprise/ADC mode (see above — that path doesn't support image generation at all right now). The fix is on Google's side: either resolve the restriction on that key/project (Cloud Console → APIs & Services), or generate a fresh key from a project you know has working Interactions API access and swap it into `PONDERCANVAS_GEMINI_IMAGE_API_KEY`.
