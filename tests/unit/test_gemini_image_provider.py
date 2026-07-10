import base64

import pytest

from pondercanvas.providers.image.gemini_image import GeminiImageProvider


class _FakeOutputImage:
    def __init__(self, data: str | None, mime_type: str | None):
        self.data = data
        self.mime_type = mime_type


class _FakeContentBlock:
    def __init__(self, text: str | None = None):
        self.text = text


class _FakeStep:
    def __init__(self, content: list[_FakeContentBlock]):
        self.content = content


class _FakeInteraction:
    def __init__(
        self,
        output_image: _FakeOutputImage | None = None,
        interaction_id: str = "interaction-id",
        status: str = "completed",
        steps: list[_FakeStep] | None = None,
        output_text: str | None = None,
    ):
        self.output_image = output_image
        self.id = interaction_id
        self.status = status
        self.steps = steps if steps is not None else []
        self.output_text = output_text


class _FakeInteractionsResource:
    def __init__(self, interaction: _FakeInteraction, calls: list[dict]):
        self._interaction = interaction
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        return self._interaction


class _FakeGenaiClient:
    def __init__(self, interaction: _FakeInteraction, calls: list[dict], **kwargs):
        self.interactions = _FakeInteractionsResource(interaction, calls)
        self.init_kwargs = kwargs


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _install_fake_client(monkeypatch, interaction, calls, client_init_calls=None):
    def factory(**kwargs):
        if client_init_calls is not None:
            client_init_calls.append(kwargs)
        return _FakeGenaiClient(interaction, calls, **kwargs)

    monkeypatch.setattr("pondercanvas.providers.image.gemini_image.genai.Client", factory)


class TestGeminiImageProviderGenerate:
    def test_extracts_output_image_bytes(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"png-bytes"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="gemini-image-x", api_key="k")
        result = provider.generate("draw a cat", [])

        assert result.image_bytes == b"png-bytes"
        assert result.mime_type == "image/png"
        assert result.provider == "gemini"
        assert result.model_id == "gemini-image-x"
        assert result.interaction_id == "interaction-id"

    def test_uses_configured_model_id_in_call(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="gemini-image-y", api_key="k")
        provider.generate("prompt", [])

        assert calls[0]["model"] == "gemini-image-y"

    def test_includes_reference_images_and_prompt_in_input(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        reference = b"\x89PNG\r\n\x1a\nfake-ref-bytes"
        provider.generate("draw a cat", [reference])

        input_content = calls[0]["input"]
        assert len(input_content) == 2  # one reference image block + the text block
        assert input_content[0]["type"] == "image"
        assert input_content[0]["data"] == _b64(reference)
        assert input_content[-1] == {"type": "text", "text": "draw a cat"}

    def test_aspect_ratio_override_via_params(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", aspect_ratio="1:1")
        provider.generate("prompt", [], aspect_ratio="16:9")

        assert calls[0]["response_format"]["aspect_ratio"] == "16:9"

    def test_default_aspect_ratio_used_when_not_overridden(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", aspect_ratio="4:3")
        provider.generate("prompt", [])

        assert calls[0]["response_format"]["aspect_ratio"] == "4:3"

    def test_response_modalities_is_image_only_not_text(self, monkeypatch):
        # Must NOT include text: with text allowed, these models will return a
        # prose description of the image instead of drawing it. Lowercase --
        # the API 400s on uppercase "IMAGE" (confirmed live 2026-07-10).
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])

        assert calls[0]["response_modalities"] == ["image"]

    def test_mime_type_omitted_from_response_format_by_default(self, monkeypatch):
        # ImageResponseFormat.mime_type only ever accepts "image/jpeg" in this
        # SDK -- there is no way to explicitly request PNG, so it must be
        # omitted (letting the API default apply) for the default png setting.
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])

        assert "mime_type" not in calls[0]["response_format"]

    def test_mime_type_included_when_output_mime_type_is_jpeg(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/jpeg"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", output_mime_type="image/jpeg")
        provider.generate("prompt", [])

        assert calls[0]["response_format"]["mime_type"] == "image/jpeg"

    def test_previous_interaction_id_passed_through_when_provided(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [], previous_interaction_id="prev-123")

        assert calls[0]["previous_interaction_id"] == "prev-123"

    def test_previous_interaction_id_is_none_when_not_provided(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])

        assert calls[0]["previous_interaction_id"] is None

    def test_google_image_search_tool_included_by_default(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])

        assert calls[0]["tools"] == [
            {"type": "google_search", "search_types": ["web_search", "image_search"]}
        ]

    def test_google_image_search_tool_omitted_when_disabled(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", image_search_enabled=False)
        provider.generate("prompt", [])

        assert calls[0]["tools"] is None

    def test_raises_when_no_output_image_in_response(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=None)
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="no output image"):
            provider.generate("prompt", [])

    def test_error_includes_status_on_incomplete_interaction(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=None, status="failed")
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="status=failed"):
            provider.generate("prompt", [])

    def test_error_includes_refusal_text_from_steps(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(
            output_image=None,
            steps=[_FakeStep([_FakeContentBlock(text="I can't create that image.")])],
        )
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="I can't create that image."):
            provider.generate("prompt", [])

    def test_error_includes_output_text_when_no_steps(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=None, output_text="Sorry, I can't do that.")
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="Sorry, I can't do that."):
            provider.generate("prompt", [])

    def test_error_notes_no_diagnostic_info_when_response_is_empty(self, monkeypatch):
        calls: list[dict] = []
        interaction = _FakeInteraction(output_image=None)
        _install_fake_client(monkeypatch, interaction, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="no additional diagnostic info"):
            provider.generate("prompt", [])

    def test_client_is_constructed_with_api_key(self, monkeypatch):
        calls: list[dict] = []
        init_calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="my-secret-key")
        provider.generate("prompt", [])

        assert init_calls[0]["api_key"] == "my-secret-key"

    def test_client_is_cached_across_calls(self, monkeypatch):
        calls: list[dict] = []
        init_calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])
        provider.generate("prompt", [])

        assert len(init_calls) == 1

    def test_enterprise_defaults_to_false(self, monkeypatch):
        calls: list[dict] = []
        init_calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])

        assert init_calls[0]["enterprise"] is False

    def test_enterprise_true_raises_instead_of_hitting_the_unsupported_endpoint(self, monkeypatch):
        # The Interactions API isn't deployed on the Enterprise/Vertex AI
        # endpoint yet -- it 404s at the host level rather than working. Fail
        # fast with an actionable message instead of letting that opaque 404
        # surface from deep inside the SDK.
        calls: list[dict] = []
        init_calls: list[dict] = []
        interaction = _FakeInteraction(output_image=_FakeOutputImage(_b64(b"x"), "image/png"))
        _install_fake_client(monkeypatch, interaction, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", enterprise=True)
        with pytest.raises(RuntimeError, match="Enterprise/Vertex AI endpoint"):
            provider.generate("prompt", [])

        assert init_calls == []
