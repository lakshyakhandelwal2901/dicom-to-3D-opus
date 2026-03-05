"""
medrecon_engine.core.dataset_scanner
======================================
Scan a directory tree and discover valid DICOM series.

Responsibilities
----------------
1.  Recursively walk *root_path* and collect candidate files.
2.  Group files by SeriesInstanceUID.
3.  For each group, record metadata (modality, description, # slices).
4.  Return an ordered list of ``SeriesInfo`` objects for the caller to
    choose from — or auto‑select the largest CT series.

This module never loads pixel data — it only reads DICOM headers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset

from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


@dataclass
class SeriesInfo:
    """Lightweight descriptor for one DICOM series."""

    series_uid: str
    modality: str = ""
    series_description: str = ""
    patient_id: str = ""
    study_description: str = ""
    manufacturer: str = ""
    num_files: int = 0
    file_paths: list[Path] = field(default_factory=list)

    @property
    def is_ct(self) -> bool:
        return self.modality.upper() == "CT"


class DatasetScanner:
    """Discover and catalogue DICOM series on disk."""

    DICOM_EXTENSIONS: set[str] = {"", ".dcm", ".dicom", ".ima"}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def scan(self, root_path: str | Path) -> list[SeriesInfo]:
        """Scan *root_path* and return a list of discovered series.

        The list is sorted by number of files (largest first).
        """
        root = Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")

        candidates = self._collect_candidates(root)
        log.info("Scanner: found %d candidate files under %s", len(candidates), root)

        series_map: dict[str, SeriesInfo] = {}

        for fpath in candidates:
            try:
                ds = pydicom.dcmread(str(fpath), stop_before_pixels=True, force=True)
            except Exception:
                continue

            uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            if uid not in series_map:
                series_map[uid] = SeriesInfo(
                    series_uid=uid,
                    modality=str(getattr(ds, "Modality", "")),
                    series_description=str(getattr(ds, "SeriesDescription", "")),
                    patient_id=str(getattr(ds, "PatientID", "")),
                    study_description=str(getattr(ds, "StudyDescription", "")),
                    manufacturer=str(getattr(ds, "Manufacturer", "")),
                )
            info = series_map[uid]
            info.num_files += 1
            info.file_paths.append(fpath)

        result = sorted(series_map.values(), key=lambda s: s.num_files, reverse=True)
        log.info("Scanner: discovered %d series (%d are CT)",
                 len(result), sum(1 for s in result if s.is_ct))
        return result

    def select_best_ct(self, root_path: str | Path) -> SeriesInfo:
        """Convenience: scan and return the largest CT series.

        Raises ``ValueError`` if no CT series is found.
        """
        all_series = self.scan(root_path)
        ct_series = [s for s in all_series if s.is_ct]
        if not ct_series:
            raise ValueError(f"No CT series found under {root_path}")
        best = ct_series[0]  # already sorted largest-first
        log.info("Scanner: selected series %s (%d slices, '%s')",
                 best.series_uid[:16], best.num_files, best.series_description)
        return best

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #
    def _collect_candidates(self, root: Path) -> list[Path]:
        """Recursively collect files that look like DICOM."""
        candidates: list[Path] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in self.DICOM_EXTENSIONS:
                    candidates.append(fpath)
        return candidates
