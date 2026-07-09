import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pondercanvas.config.constants import REDACTED_MARKER, SECRET_FIELD_SUFFIX
from pondercanvas.schemas.trace import RunTrace

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_FILENAME = "pondercanvas.log"
_MAX_LOG_BYTES = 5_000_000
_LOG_BACKUP_COUNT = 3


def configure_logging(output_dir: Path, level: int = logging.INFO) -> None:
    """Routes every `pondercanvas.*` logger -- including uncaught exceptions
    from a Generate request -- to output_dir/pondercanvas.log (rotated at 5MB,
    3 backups kept), in addition to the console. Idempotent: safe to call more
    than once (e.g. across repeated app restarts in the same process)."""
    log_path = output_dir / _LOG_FILENAME
    logger = logging.getLogger("pondercanvas")
    logger.setLevel(level)

    already_configured = any(
        isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path.resolve())
        for h in logger.handlers
    )
    if already_configured:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_LOG_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def redact_secrets(value: Any) -> Any:
    """Recursively masks any dict key ending in _api_key. Defense in depth:
    RunTrace.settings_snapshot is already built via EffectiveSettings.redacted(),
    but this catches any secret that ends up elsewhere in the record."""
    if isinstance(value, dict):
        return {
            key: (
                REDACTED_MARKER
                if key.endswith(SECRET_FIELD_SUFFIX) and val is not None
                else redact_secrets(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def log_run_trace(trace: RunTrace, log_path: Path) -> None:
    """Appends one JSONL record for this run to log_path, creating parent
    directories as needed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = redact_secrets(trace.model_dump(mode="json"))
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
