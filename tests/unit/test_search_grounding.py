from pondercanvas.providers.search.gemini_search import ground_with_search


class _FakeWeb:
    def __init__(self, uri, title=None):
        self.uri = uri
        self.title = title


class _FakeChunk:
    def __init__(self, web=None):
        self.web = web


class _FakeGroundingMetadata:
    def __init__(self, chunks):
        self.grounding_chunks = chunks


class _FakeCandidate:
    def __init__(self, grounding_metadata):
        self.grounding_metadata = grounding_metadata


class _FakeResponse:
    def __init__(self, text, candidates):
        self.text = text
        self.candidates = candidates


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


def _install_fake_client(monkeypatch, response, calls):
    def factory(**kwargs):
        return _FakeGenaiClient(response, calls, **kwargs)

    monkeypatch.setattr("pondercanvas.providers.search.gemini_search.genai.Client", factory)


class TestGroundWithSearch:
    def test_extracts_summary_text_and_citations(self, monkeypatch):
        calls: list[dict] = []
        chunks = [
            _FakeChunk(web=_FakeWeb(uri="https://example.com/a", title="A")),
            _FakeChunk(web=_FakeWeb(uri="https://example.com/b", title="B")),
        ]
        response = _FakeResponse(
            text="grounded summary", candidates=[_FakeCandidate(_FakeGroundingMetadata(chunks))]
        )
        _install_fake_client(monkeypatch, response, calls)

        result = ground_with_search(["watercolor bicycle"], "k", "model-x")

        assert result.summary_text == "grounded summary"
        assert result.queries_used == ["watercolor bicycle"]
        assert [c.url for c in result.citations] == ["https://example.com/a", "https://example.com/b"]
        assert [c.title for c in result.citations] == ["A", "B"]

    def test_skips_chunks_without_web_or_uri(self, monkeypatch):
        calls: list[dict] = []
        chunks = [_FakeChunk(web=None), _FakeChunk(web=_FakeWeb(uri=None))]
        response = _FakeResponse(
            text="summary", candidates=[_FakeCandidate(_FakeGroundingMetadata(chunks))]
        )
        _install_fake_client(monkeypatch, response, calls)

        result = ground_with_search(["q"], "k", "model-x")

        assert result.citations == []

    def test_no_candidates_returns_empty_citations(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(text="summary", candidates=[])
        _install_fake_client(monkeypatch, response, calls)

        result = ground_with_search(["q"], "k", "model-x")

        assert result.citations == []
        assert result.summary_text == "summary"

    def test_uses_google_search_grounding_tool(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(text="s", candidates=[])
        _install_fake_client(monkeypatch, response, calls)

        ground_with_search(["q"], "k", "model-x")

        config = calls[0]["config"]
        assert config.tools[0].google_search is not None

    def test_combines_multiple_queries_into_one_call(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(text="s", candidates=[])
        _install_fake_client(monkeypatch, response, calls)

        ground_with_search(["q1", "q2"], "k", "model-x")

        assert len(calls) == 1
        assert "q1" in calls[0]["contents"]
        assert "q2" in calls[0]["contents"]

    def test_empty_text_becomes_empty_string_not_none(self, monkeypatch):
        calls: list[dict] = []
        response = _FakeResponse(text=None, candidates=[])
        _install_fake_client(monkeypatch, response, calls)

        result = ground_with_search(["q"], "k", "model-x")

        assert result.summary_text == ""
