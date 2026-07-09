import logging
from io import BytesIO

import pytest
from PIL import Image

from pondercanvas.providers.scoring.siglip import SiglipScorer

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


def _real_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buf, format="PNG")
    return buf.getvalue()


def _break(monkeypatch, cls_name: str) -> None:
    """Makes transformers.<cls_name>.from_pretrained raise, regardless of
    whether the real 'siglip' extra is actually installed in this
    environment -- these tests must be deterministic either way."""

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(getattr(transformers, cls_name), "from_pretrained", _raise)


class TestSiglipScorerUnavailable:
    def test_score_returns_none_instead_of_raising(self, monkeypatch, caplog):
        _break(monkeypatch, "SiglipModel")
        scorer = SiglipScorer()

        with caplog.at_level(logging.WARNING):
            result = scorer.score(b"not-a-real-image", "a cat")

        assert result is None

    def test_logs_a_warning_mentioning_siglip(self, monkeypatch, caplog):
        _break(monkeypatch, "SiglipModel")
        scorer = SiglipScorer()

        with caplog.at_level(logging.WARNING):
            scorer.score(b"not-a-real-image", "a cat")

        assert any("siglip" in record.message.lower() for record in caplog.records)

    def test_repeated_calls_keep_returning_none_without_raising(self, monkeypatch, caplog):
        _break(monkeypatch, "SiglipModel")
        scorer = SiglipScorer()

        with caplog.at_level(logging.WARNING):
            first = scorer.score(b"image-one", "a cat")
            second = scorer.score(b"image-two", "a dog")

        assert first is None
        assert second is None

    def test_failure_is_cached_so_it_only_logs_once(self, monkeypatch, caplog):
        _break(monkeypatch, "SiglipModel")
        scorer = SiglipScorer()

        with caplog.at_level(logging.WARNING):
            scorer.score(b"image-one", "a cat")
            count_after_first = len(caplog.records)
            scorer.score(b"image-two", "a dog")
            count_after_second = len(caplog.records)

        assert count_after_first > 0
        assert count_after_second == count_after_first

    def test_partial_failure_does_not_leave_stale_model_without_processor(self, monkeypatch, caplog):
        # Regression test: SiglipModel loads fine but SiglipProcessor doesn't
        # (e.g. a missing tokenizer backend like sentencepiece). The first
        # call must disable the scorer entirely rather than leaving _model
        # set with _processor still None, which would make the next call's
        # "already loaded" fast path wrongly report ready and then crash.
        _break(monkeypatch, "SiglipProcessor")
        scorer = SiglipScorer()

        with caplog.at_level(logging.WARNING):
            first = scorer.score(b"image-one", "a cat")
            second = scorer.score(b"image-two", "a dog")

        assert first is None
        assert second is None
        assert scorer._model is None
        assert scorer._processor is None


class _FakeBatchEncoding(dict):
    def to(self, *args, **kwargs):
        return self


class _FakeProcessor:
    def __call__(self, text, images, return_tensors, padding):
        return _FakeBatchEncoding()


class _FakeModelOutput:
    def __init__(self, logits_per_image):
        self.logits_per_image = logits_per_image


class _FakeModel:
    def __call__(self, **inputs):
        return _FakeModelOutput(torch.tensor([[2.0]]))


class TestSiglipScorerSuccess:
    def test_score_returns_sigmoid_of_logits_per_image(self, monkeypatch):
        monkeypatch.setattr(transformers.SiglipModel, "from_pretrained", lambda *a, **k: _FakeModel())
        monkeypatch.setattr(
            transformers.SiglipProcessor, "from_pretrained", lambda *a, **k: _FakeProcessor()
        )
        scorer = SiglipScorer()

        result = scorer.score(_real_png_bytes(), "a cat")

        assert result == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())

    def test_model_and_processor_are_cached_across_calls(self, monkeypatch):
        model_calls = []
        processor_calls = []
        monkeypatch.setattr(
            transformers.SiglipModel,
            "from_pretrained",
            lambda *a, **k: model_calls.append(1) or _FakeModel(),
        )
        monkeypatch.setattr(
            transformers.SiglipProcessor,
            "from_pretrained",
            lambda *a, **k: processor_calls.append(1) or _FakeProcessor(),
        )
        scorer = SiglipScorer()

        scorer.score(_real_png_bytes(), "a cat")
        scorer.score(_real_png_bytes(), "a dog")

        assert len(model_calls) == 1
        assert len(processor_calls) == 1
