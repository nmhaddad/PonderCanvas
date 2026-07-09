import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

import pytest

from pondercanvas.logging_utils import configure_logging, log_run_trace, redact_secrets
from pondercanvas.schemas.trace import RunTrace
from tests.fixtures.sample_brief import sample_brief


def _trace(**overrides) -> RunTrace:
    defaults: dict = dict(
        run_id="r1",
        brief=sample_brief(),
        iterations=[],
        passed=True,
        stopped_reason="passed",
        settings_snapshot={"chat_provider": "gemini", "google_api_key": "***REDACTED***"},
        created_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return RunTrace(**defaults)


class TestRedactSecrets:
    def test_masks_top_level_api_key(self):
        result = redact_secrets({"google_api_key": "sk-real-secret"})
        assert result["google_api_key"] == "***REDACTED***"

    def test_masks_nested_api_key(self):
        result = redact_secrets({"settings_snapshot": {"openai_api_key": "sk-real"}})
        assert result["settings_snapshot"]["openai_api_key"] == "***REDACTED***"

    def test_masks_api_key_inside_list_of_dicts(self):
        result = redact_secrets({"items": [{"stability_api_key": "sk-real"}]})
        assert result["items"][0]["stability_api_key"] == "***REDACTED***"

    def test_leaves_none_api_key_as_none(self):
        result = redact_secrets({"anthropic_api_key": None})
        assert result["anthropic_api_key"] is None

    def test_leaves_non_secret_fields_untouched(self):
        result = redact_secrets({"chat_provider": "gemini", "max_iterations": 5})
        assert result == {"chat_provider": "gemini", "max_iterations": 5}


class TestLogRunTrace:
    def test_appends_jsonl_record(self, tmp_path):
        log_path = tmp_path / "runs.jsonl"
        log_run_trace(_trace(run_id="r1"), log_path)
        log_run_trace(_trace(run_id="r2"), log_path)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["run_id"] == "r1"
        assert json.loads(lines[1])["run_id"] == "r2"

    def test_creates_parent_directory(self, tmp_path):
        log_path = tmp_path / "nested" / "dir" / "runs.jsonl"
        log_run_trace(_trace(), log_path)
        assert log_path.exists()

    def test_redacts_settings_snapshot_secrets_in_written_record(self, tmp_path):
        log_path = tmp_path / "runs.jsonl"
        trace = _trace(settings_snapshot={"google_api_key": "sk-should-never-be-written"})
        log_run_trace(trace, log_path)

        content = log_path.read_text()
        assert "sk-should-never-be-written" not in content
        assert "***REDACTED***" in content


@pytest.fixture
def _clean_pondercanvas_logger():
    """configure_logging() mutates the process-wide `pondercanvas` logger;
    snapshot and restore its handlers so tests don't leak state into each
    other (or into the real app if run in the same process)."""
    logger = logging.getLogger("pondercanvas")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    yield logger
    for handler in logger.handlers:
        if handler not in original_handlers:
            handler.close()
    logger.handlers = original_handlers
    logger.level = original_level


class TestConfigureLogging:
    def test_creates_log_file(self, tmp_path, _clean_pondercanvas_logger):
        configure_logging(tmp_path)
        assert (tmp_path / "pondercanvas.log").exists()

    def test_records_from_child_loggers_are_written_to_the_file(
        self, tmp_path, _clean_pondercanvas_logger
    ):
        configure_logging(tmp_path)
        logging.getLogger("pondercanvas.providers.scoring.siglip").warning("hello from a submodule")

        content = (tmp_path / "pondercanvas.log").read_text()
        assert "hello from a submodule" in content

    def test_calling_twice_does_not_duplicate_handlers(self, tmp_path, _clean_pondercanvas_logger):
        configure_logging(tmp_path)
        configure_logging(tmp_path)

        file_handlers = [
            h for h in _clean_pondercanvas_logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_calling_twice_does_not_duplicate_log_lines(self, tmp_path, _clean_pondercanvas_logger):
        configure_logging(tmp_path)
        configure_logging(tmp_path)
        logging.getLogger("pondercanvas.foo").warning("only once please")

        content = (tmp_path / "pondercanvas.log").read_text()
        assert content.count("only once please") == 1
