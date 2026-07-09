import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pondercanvas.schemas.brief import GenerationBrief
from pondercanvas.schemas.evaluation import EvaluationResult
from pondercanvas.schemas.grounding import GroundingResult, PhotoAttribution, SourceCitation
from pondercanvas.schemas.trace import IterationTrace, RunTrace


def _scores(value: float) -> dict:
    return {
        "prompt_adherence": value,
        "aesthetic_quality": value,
        "technical_quality": value,
        "reference_alignment": value,
    }


def _sample_brief() -> GenerationBrief:
    return GenerationBrief(
        subject="a red bicycle",
        style="watercolor",
        composition="centered, three-quarter view",
        mood="cheerful",
        palette="warm reds and oranges",
        constraints=["no text", "square crop"],
        notes_from_references="reference shows a vintage frame",
        search_queries=["watercolor bicycle illustration"],
        aspect_ratio="1:1",
        raw_user_prompt="draw me a red bicycle",
    )


class TestGenerationBrief:
    def test_round_trips_through_json(self):
        brief = _sample_brief()
        restored = GenerationBrief.model_validate_json(brief.model_dump_json())
        assert restored == brief

    def test_defaults_for_optional_fields(self):
        brief = GenerationBrief(
            subject="s",
            style="s",
            composition="s",
            mood="s",
            palette="s",
            raw_user_prompt="s",
        )
        assert brief.constraints == []
        assert brief.search_queries == []
        assert brief.notes_from_references is None
        assert brief.aspect_ratio == "1:1"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            GenerationBrief(style="s", composition="s", mood="s", palette="s", raw_user_prompt="s")


class TestGroundingResult:
    def test_round_trips_with_citations_and_attributions(self):
        result = GroundingResult(
            queries_used=["q1", "q2"],
            summary_text="grounded context",
            citations=[
                SourceCitation(url="https://example.com", title="Example", snippet="a snippet")
            ],
            downloaded_reference_count=1,
            photo_attributions=[
                PhotoAttribution(
                    photographer_name="Alice",
                    photographer_profile_url="https://unsplash.com/@alice",
                    photo_page_url="https://unsplash.com/photos/p1",
                )
            ],
        )
        restored = GroundingResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_defaults_are_empty(self):
        result = GroundingResult()
        assert result.queries_used == []
        assert result.citations == []
        assert result.downloaded_reference_count == 0
        assert result.photo_attributions == []


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


class TestEvaluationResultSchemaIsMldevCompatible:
    def test_json_schema_never_emits_additional_properties_key(self):
        # Gemini's Developer API (mldev) structured output rejects any
        # `additionalProperties` keyword outright (client-side ValueError,
        # before any network call) -- which is exactly what a free-form
        # dict[str, float] field emits. scores must stay a fixed-shape
        # object (CriterionScores), not a dict, to keep working there.
        assert not _contains_key(EvaluationResult.model_json_schema(), "additionalProperties")


class TestEvaluationResult:
    def test_parses_gemini_style_json_with_pass_keyword(self):
        # Gemini's structured JSON output uses the literal key "pass" (a
        # Python keyword).
        raw = json.dumps(
            {
                "scores": {
                    "prompt_adherence": 4.5,
                    "aesthetic_quality": 4.5,
                    "technical_quality": 4.0,
                    "reference_alignment": 4.0,
                },
                "overall": 4.25,
                "pass": True,
                "feedback": "Looks good",
                "threshold": 4.0,
            }
        )
        result = EvaluationResult.model_validate_json(raw)
        assert result.is_passing is True
        assert result.overall == 4.25
        assert result.scores.aesthetic_quality == 4.5

    def test_can_also_be_constructed_by_field_name(self):
        # populate_by_name=True: Python call sites use is_passing=, not pass=.
        result = EvaluationResult(
            scores=_scores(5.0),
            overall=5.0,
            is_passing=True,
            feedback="great",
            threshold=4.0,
        )
        assert result.is_passing is True

    def test_dump_by_alias_emits_pass_key(self):
        result = EvaluationResult(
            scores=_scores(5.0), overall=5.0, is_passing=False, feedback="meh", threshold=4.0
        )
        dumped = result.model_dump(by_alias=True)
        assert dumped["pass"] is False
        assert "is_passing" not in dumped


class TestRunTrace:
    def test_round_trips_full_trace(self):
        now = datetime.now(UTC)
        evaluation = EvaluationResult(
            scores=_scores(4.0), overall=4.0, is_passing=True, feedback="ok", threshold=4.0
        )
        iteration = IterationTrace(
            iteration_index=0,
            prompt_used="draw a red bicycle, watercolor style",
            image_path="/tmp/out/0.png",
            evaluation=evaluation,
            created_at=now,
        )
        trace = RunTrace(
            run_id="run-1",
            brief=_sample_brief(),
            grounding=GroundingResult(summary_text="ctx"),
            iterations=[iteration],
            final_image_path="/tmp/out/0.png",
            passed=True,
            stopped_reason="passed",
            settings_snapshot={"chat_provider": "gemini"},
            created_at=now,
        )
        restored = RunTrace.model_validate_json(trace.model_dump_json())
        assert restored == trace

    def test_stopped_reason_is_constrained(self):
        with pytest.raises(ValidationError):
            RunTrace(
                run_id="run-1",
                brief=_sample_brief(),
                stopped_reason="not-a-real-reason",
                created_at=datetime.now(UTC),
            )
