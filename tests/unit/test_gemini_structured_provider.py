import json

from pydantic import BaseModel

from pondercanvas.providers.structured.gemini_structured import GeminiStructuredVisionProvider


class _SampleSchema(BaseModel):
    title: str
    score: float


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


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


def _install_fake_client(monkeypatch, response_text, calls):
    def factory(**kwargs):
        return _FakeGenaiClient(_FakeResponse(response_text), calls, **kwargs)

    monkeypatch.setattr(
        "pondercanvas.providers.structured.gemini_structured.genai.Client", factory
    )


class TestGeminiStructuredVisionProvider:
    def test_parses_response_into_given_schema(self, monkeypatch):
        calls: list[dict] = []
        raw = json.dumps({"title": "a design concept", "score": 4.5})
        _install_fake_client(monkeypatch, raw, calls)

        provider = GeminiStructuredVisionProvider(model_id="m", api_key="k")
        result = provider.generate_structured("describe this", [], _SampleSchema)

        assert isinstance(result, _SampleSchema)
        assert result.title == "a design concept"
        assert result.score == 4.5

    def test_config_requests_json_mime_and_schema(self, monkeypatch):
        calls: list[dict] = []
        raw = json.dumps({"title": "t", "score": 1.0})
        _install_fake_client(monkeypatch, raw, calls)

        provider = GeminiStructuredVisionProvider(model_id="m", api_key="k")
        provider.generate_structured("prompt", [], _SampleSchema)

        config = calls[0]["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is _SampleSchema

    def test_uses_configured_model_id(self, monkeypatch):
        calls: list[dict] = []
        raw = json.dumps({"title": "t", "score": 1.0})
        _install_fake_client(monkeypatch, raw, calls)

        provider = GeminiStructuredVisionProvider(model_id="structured-model-x", api_key="k")
        provider.generate_structured("prompt", [], _SampleSchema)

        assert calls[0]["model"] == "structured-model-x"

    def test_includes_images_and_prompt_in_contents(self, monkeypatch):
        calls: list[dict] = []
        raw = json.dumps({"title": "t", "score": 1.0})
        _install_fake_client(monkeypatch, raw, calls)

        provider = GeminiStructuredVisionProvider(model_id="m", api_key="k")
        image_bytes = b"\x89PNG\r\n\x1a\nfake"
        provider.generate_structured("prompt", [image_bytes], _SampleSchema)

        contents = calls[0]["contents"]
        assert contents[-1] == "prompt"
        assert len(contents) == 2

    def test_no_images_still_includes_prompt(self, monkeypatch):
        calls: list[dict] = []
        raw = json.dumps({"title": "t", "score": 1.0})
        _install_fake_client(monkeypatch, raw, calls)

        provider = GeminiStructuredVisionProvider(model_id="m", api_key="k")
        provider.generate_structured("just a prompt", [], _SampleSchema)

        assert calls[0]["contents"] == ["just a prompt"]
