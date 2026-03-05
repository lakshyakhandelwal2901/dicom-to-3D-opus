"""
medrecon_engine.audit.logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Structured audit-grade logging for the entire MedRecon pipeline.

* **get_logger(name)** — returns a stdlib `logging.Logger` pre-configured
  with a rich, colour-coded console handler *and* a rotating JSON-lines
  file handler suitable for compliance archives.
* **AuditRecord** — frozen dataclass written to the JSONL audit trail
  after every pipeline run.
* **write_audit_record / read_audit_trail** — convenience I/O helpers.

Design goals
------------
1. Deterministic — no randomness, timestamps use UTC ISO-8601.
2. On-prem safe — everything is local files; no cloud telemetry.
3. Compliant — JSONL format is ingestible by SIEM / ELK / Splunk.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import platform
import socket
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_LOG_DIR = Path("logs")
_AUDIT_FILENAME = "medrecon_audit.jsonl"
_APP_LOG_FILENAME = "medrecon_engine.log"
_MAX_LOG_BYTES = 50 * 1024 * 1024  # 50 MB per file
_BACKUP_COUNT = 10
_DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
_CONSOLE_FMT = "%(asctime)s │ %(levelname)-8s │ %(name)-35s │ %(message)s"
_FILE_FMT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)


class _UTCFormatter(logging.Formatter):
    """Formatter that always emits UTC timestamps."""

    converter = lambda *_: datetime.now(timezone.utc).timetuple()  # noqa: E731


# ---------------------------------------------------------------------------
# Singleton handler cache (prevents duplicate handlers on repeated calls)
# ---------------------------------------------------------------------------
_CONFIGURED: Dict[str, bool] = {}


def _ensure_log_dir(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_logger(
    name: str,
    *,
    level: int = logging.DEBUG,
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Return a fully-configured :class:`logging.Logger`.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int
        Minimum severity.  Default ``DEBUG`` (file captures everything;
        console shows INFO+).
    log_dir : Path | None
        Directory for log files.  Defaults to ``./logs``.
    console : bool
        Attach a coloured ``StreamHandler`` to *stderr*.
    file : bool
        Attach a ``RotatingFileHandler`` writing to *log_dir*.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler ────────────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(_UTCFormatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
        logger.addHandler(ch)

    # ── File handler ───────────────────────────────────────────────────
    if file:
        _dir = _ensure_log_dir(log_dir or _DEFAULT_LOG_DIR)
        fh = logging.handlers.RotatingFileHandler(
            _dir / _APP_LOG_FILENAME,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_UTCFormatter(_FILE_FMT, datefmt=_DATE_FMT))
        logger.addHandler(fh)

    _CONFIGURED[name] = True
    return logger


# ---------------------------------------------------------------------------
# Audit Record — immutable, JSON-serialisable
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuditRecord:
    """One record per pipeline invocation, written to JSONL audit trail."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: platform.python_version())
    engine_version: str = "1.0.0"

    # Pipeline inputs
    dicom_path: str = ""
    anatomy: str = ""
    num_slices: int = 0
    voxel_spacing_mm: tuple = ()

    # Pipeline outputs
    output_stl_path: str = ""
    mesh_vertices: int = 0
    mesh_faces: int = 0
    volume_cm3: float = 0.0

    # Quality
    confidence_score: float = 0.0
    confidence_grade: str = ""
    topology_valid: bool = False
    manifold: bool = False

    # HU model
    hu_adaptive_min: float = 0.0
    hu_adaptive_max: float = 0.0

    # Timing
    elapsed_seconds: float = 0.0

    # Errors
    success: bool = True
    error_message: str = ""

    # Free-form metadata
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audit I/O helpers
# ---------------------------------------------------------------------------
def write_audit_record(
    record: AuditRecord,
    *,
    audit_dir: Optional[Path] = None,
) -> Path:
    """Append *record* as a single JSON line to the audit file.

    Returns the path of the audit file.
    """
    _dir = _ensure_log_dir(audit_dir or _DEFAULT_LOG_DIR)
    path = _dir / _AUDIT_FILENAME

    data = asdict(record)
    # Ensure tuple serialises as list for JSON
    data["voxel_spacing_mm"] = list(data.get("voxel_spacing_mm", []))

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, default=str) + "\n")

    return path


def read_audit_trail(
    *,
    audit_dir: Optional[Path] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Read back the JSONL audit trail.

    Parameters
    ----------
    audit_dir : Path | None
        Directory containing the audit JSONL file.
    limit : int
        Max records to return (0 = all, newest-first).
    """
    path = (audit_dir or _DEFAULT_LOG_DIR) / _AUDIT_FILENAME
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records.reverse()  # newest first
    if limit > 0:
        records = records[:limit]
    return records


# ---------------------------------------------------------------------------
# Module-level convenience logger
# ---------------------------------------------------------------------------
_log = get_logger(__name__)
