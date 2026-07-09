import pytest

from pondercanvas.providers.image.gemini_image import GeminiImageProvider


class _FakeInlineData:
    def __init__(self, data: bytes, mime_type: str):
        self.data = data
        self.mime_type = mime_type


class _FakePart:
    def __init__(self, inline_data=None, text=None):
        self.inline_data = inline_data
        self.text = text


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts, finish_reason=None, finish_message=None):
        self.content = _FakeContent(parts)
        self.finish_reason = finish_reason
        self.finish_message = finish_message


class _FakePromptFeedback:
    def __init__(self, block_reason=None):
        self.block_reason = block_reason


class _FakeResponse:
    def __init__(self, parts=None, candidates=None, prompt_feedback=None):
        if candidates is not None:
            self.candidates = candidates
        else:
            self.candidates = [_FakeCandidate(parts or [])]
        self.prompt_feedback = prompt_feedback


class _FakeModels:
    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    def generate_content(self, **kwargs):
        self._calls.append(kwargs)
        return self._response


class _FakeGenaiClient:
    def __init__(self, response, calls, **kwargs):
        self.models = _FakeModels(response, calls)
        self.init_kwargs = kwargs


def _install_fake_client(monkeypatch, response, calls, client_init_calls=None):
    def factory(**kwargs):
        if client_init_calls is not None:
            client_init_calls.append(kwargs)
        return _FakeGenaiClient(response, calls, **kwargs)

    monkeypatch.setattr("pondercanvas.providers.image.gemini_image.genai.Client", factory)


class TestGeminiImageProviderGenerate:
    def test_extracts_inline_image_bytes(self, monkeypatch):
        calls: list[dict] = []
        inline_data = _FakeInlineData(b"png-bytes", "image/png")
        response = _FakeResponse([_FakePart(inline_data=inline_data)])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="gemini-image-x", api_key="k")
        result = provider.generate("draw a cat", [])

        assert result.image_bytes == b"png-bytes"
        assert result.mime_type == "image/png"
        assert result.provider == "gemini"
        assert result.model_id == "gemini-image-x"

    def test_uses_configured_model_id_in_call(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="gemini-image-y", api_key="k")
        provider.generate("prompt", [])

        assert calls[0]["model"] == "gemini-image-y"

    def test_includes_reference_images_and_prompt_in_contents(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        reference = b"\x89PNG\r\n\x1a\nfake-ref-bytes"
        provider.generate("draw a cat", [reference])

        contents = calls[0]["contents"]
        assert contents[-1] == "draw a cat"
        assert len(contents) == 2  # one reference Part + the prompt string

    def test_aspect_ratio_override_via_params(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", aspect_ratio="1:1")
        provider.generate("prompt", [], aspect_ratio="16:9")

        assert calls[0]["config"].image_config.aspect_ratio == "16:9"

    def test_default_aspect_ratio_used_when_not_overridden(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", aspect_ratio="4:3")
        provider.generate("prompt", [])

        assert calls[0]["config"].image_config.aspect_ratio == "4:3"

    def test_response_modalities_is_image_only_not_text(self, monkeypatch):
        # Must NOT include TEXT: with TEXT allowed, these models will return a
        # prose description of the image (finish_reason=STOP, no error)
        # instead of drawing it. Applies to both developer and enterprise mode.
        for enterprise in (False, True):
            calls: list[dict] = []
            response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
            _install_fake_client(monkeypatch, response, calls)

            provider = GeminiImageProvider(model_id="m", api_key="k", enterprise=enterprise)
            provider.generate("prompt", [])

            assert calls[0]["config"].response_modalities == ["IMAGE"]

    def test_output_mime_type_omitted_in_developer_api_mode(self, monkeypatch):
        # output_mime_type is Enterprise/Vertex-only; the real SDK raises a
        # client-side ValueError if sent while enterprise=False.
        calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", enterprise=False)
        provider.generate("prompt", [])

        assert calls[0]["config"].image_config.output_mime_type is None

    def test_output_mime_type_included_in_enterprise_mode(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(
            model_id="m", api_key="k", enterprise=True, output_mime_type="image/jpeg"
        )
        provider.generate("prompt", [])

        assert calls[0]["config"].image_config.output_mime_type == "image/jpeg"

    def test_raises_when_no_inline_data_in_response(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=None)])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="no inline image data"):
            provider.generate("prompt", [])

    def test_error_includes_finish_reason_on_missing_image(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(
            candidates=[_FakeCandidate([_FakePart()], finish_reason="IMAGE_SAFETY")]
        )
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="finish_reason=IMAGE_SAFETY"):
            provider.generate("prompt", [])

    def test_error_includes_refusal_text_instead_of_image(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(
            candidates=[_FakeCandidate([_FakePart(text="I can't create that image.")])]
        )
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="I can't create that image."):
            provider.generate("prompt", [])

    def test_error_includes_prompt_block_reason(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(
            candidates=[_FakeCandidate([])],
            prompt_feedback=_FakePromptFeedback(block_reason="SAFETY"),
        )
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="prompt blocked: SAFETY"):
            provider.generate("prompt", [])

    def test_error_notes_no_candidates_returned(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(candidates=[])
        _install_fake_client(monkeypatch, response, calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="no candidates returned"):
            provider.generate("prompt", [])

    def test_client_is_constructed_with_api_key(self, monkeypatch):
        calls: list[dict] = []
        init_calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="my-secret-key")
        provider.generate("prompt", [])

        assert init_calls[0]["api_key"] == "my-secret-key"

    def test_client_is_cached_across_calls(self, monkeypatch):
        calls: list[dict] = []
        init_calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])
        provider.generate("prompt", [])

        assert len(init_calls) == 1

    def test_enterprise_defaults_to_false(self, monkeypatch):
        calls: list[dict] = []
        init_calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="k")
        provider.generate("prompt", [])

        assert init_calls[0]["enterprise"] is False

    def test_enterprise_true_is_passed_to_client(self, monkeypatch):
        calls: list[dict] = []
        init_calls: list[dict] = []
        response = _FakeResponse([_FakePart(inline_data=_FakeInlineData(b"x", "image/png"))])
        _install_fake_client(monkeypatch, response, calls, client_init_calls=init_calls)

        provider = GeminiImageProvider(model_id="m", api_key="k", enterprise=True)
        provider.generate("prompt", [])

        assert init_calls[0]["enterprise"] is True
