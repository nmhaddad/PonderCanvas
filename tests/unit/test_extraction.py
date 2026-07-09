from pondercanvas.agent.extraction import extract_generation_brief
from pondercanvas.schemas.brief import GenerationBrief
from tests.fixtures.fake_structured_provider import FakeStructuredVisionProvider


def _extracted_result(**overrides):
    from pondercanvas.agent.extraction import _ExtractedFields

    defaults = dict(
        subject="a red bicycle",
        style="watercolor",
        composition="centered",
        mood="cheerful",
        palette="warm reds",
        constraints=["no text"],
        notes_from_references=None,
        search_queries=["red bicycle watercolor"],
    )
    defaults.update(overrides)
    return _ExtractedFields(**defaults)


class TestExtractGenerationBrief:
    def test_returns_generation_brief(self):
        provider = FakeStructuredVisionProvider([_extracted_result()])
        brief = extract_generation_brief("draw a red bicycle", [], provider)
        assert isinstance(brief, GenerationBrief)
        assert brief.subject == "a red bicycle"

    def test_raw_user_prompt_comes_from_argument_not_llm(self):
        provider = FakeStructuredVisionProvider([_extracted_result()])
        brief = extract_generation_brief("the literal user prompt", [], provider)
        assert brief.raw_user_prompt == "the literal user prompt"

    def test_aspect_ratio_comes_from_argument_not_llm(self):
        provider = FakeStructuredVisionProvider([_extracted_result()])
        brief = extract_generation_brief("prompt", [], provider, aspect_ratio="16:9")
        assert brief.aspect_ratio == "16:9"

    def test_default_aspect_ratio_is_square(self):
        provider = FakeStructuredVisionProvider([_extracted_result()])
        brief = extract_generation_brief("prompt", [], provider)
        assert brief.aspect_ratio == "1:1"

    def test_passes_reference_images_to_provider(self):
        provider = FakeStructuredVisionProvider([_extracted_result()])
        extract_generation_brief("prompt", [b"ref1", b"ref2"], provider)
        assert provider.calls[0]["images"] == [b"ref1", b"ref2"]

    def test_prompt_includes_user_request(self):
        provider = FakeStructuredVisionProvider([_extracted_result()])
        extract_generation_brief("draw a spaceship", [], provider)
        assert "draw a spaceship" in provider.calls[0]["prompt"]
