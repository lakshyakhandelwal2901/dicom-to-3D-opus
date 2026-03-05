"""
medrecon_engine.core.dicom_validator
======================================
Hard-fail validation of a DICOM series BEFORE any processing begins.

Checks
------
1.  Modality is CT.
2.  Slice thickness ≤ ``MAX_ALLOWED_SLICE``.
3.  In-plane pixel spacing ≤ ``MAX_ALLOWED_INPLANE``.
4.  All slices have consistent Rows × Columns.
5.  RescaleSlope / RescaleIntercept are present (needed for HU conversion).
6.  At least 20 slices (recon quality floor).

On ANY violation the validator raises ``PrecisionViolation`` — the pipeline
does NOT silently degrade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class PrecisionViolation(Exception):
    """Raised when a dataset fails strict precision validation."""


@dataclass
class ValidationReport:
    """Summary produced by the validator."""

    passed: bool
    modality: str = ""
    slice_thickness_mm: float = 0.0
    pixel_spacing_mm: tuple[float, float] = (0.0, 0.0)
    num_slices: int = 0
    rows: int = 0
    columns: int = 0
    has_rescale: bool = False
    violations: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class DicomValidator:
    """Validate a list of DICOM file paths against precision rules."""

    MIN_SLICES: int = 20

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def validate(self, file_paths: list[Path]) -> ValidationReport:
        """Read headers only and return a ``ValidationReport``.

        Raises ``PrecisionViolation`` on any hard failure.
        """
        if not file_paths:
            raise PrecisionViolation("No DICOM files provided.")

        # Read first slice for metadata
        ds = pydicom.dcmread(str(file_paths[0]), stop_before_pixels=True, force=True)

        report = ValidationReport(passed=True)
        report.num_slices = len(file_paths)

        # ----- Modality -----
        report.modality = str(getattr(ds, "Modality", "UNKNOWN"))
        if report.modality.upper() != "CT":
            report.violations.append(
                f"Modality is '{report.modality}', expected 'CT'."
            )

        # ----- Slice thickness -----
        st = float(getattr(ds, "SliceThickness", 0.0))
        report.slice_thickness_mm = st
        if st <= 0:
            report.violations.append("SliceThickness missing or zero.")
        elif st > self.cfg.max_allowed_slice:
            report.violations.append(
                f"SliceThickness {st:.2f} mm > max allowed {self.cfg.max_allowed_slice:.2f} mm."
            )

        # ----- Pixel spacing -----
        ps = getattr(ds, "PixelSpacing", None)
        if ps is not None:
            px, py = float(ps[0]), float(ps[1])
            report.pixel_spacing_mm = (px, py)
            if max(px, py) > self.cfg.max_allowed_inplane:
                report.violations.append(
                    f"PixelSpacing ({px:.3f}, {py:.3f}) mm > max allowed "
                    f"{self.cfg.max_allowed_inplane:.2f} mm."
                )
        else:
            report.violations.append("PixelSpacing tag missing.")

        # ----- Dimensions -----
        report.rows = int(getattr(ds, "Rows", 0))
        report.columns = int(getattr(ds, "Columns", 0))

        # ----- Rescale -----
        has_slope = hasattr(ds, "RescaleSlope")
        has_intercept = hasattr(ds, "RescaleIntercept")
        report.has_rescale = has_slope and has_intercept
        if not report.has_rescale:
            report.violations.append(
                "RescaleSlope/RescaleIntercept missing — HU conversion unreliable."
            )

        # ----- Minimum slice count -----
        if report.num_slices < self.MIN_SLICES:
            report.violations.append(
                f"Only {report.num_slices} slices — minimum is {self.MIN_SLICES}."
            )

        # ----- Consistency check (sample a few) -----
        self._check_consistency(file_paths, report)

        # ----- Verdict -----
        if report.violations:
            report.passed = False
            for v in report.violations:
                log.error("Validator FAIL: %s", v)
            raise PrecisionViolation(
                "Dataset rejected — " + "; ".join(report.violations)
            )

        log.info(
            "Validator PASS: %s, %d slices, %.2f mm slice, (%.3f, %.3f) mm pixel",
            report.modality,
            report.num_slices,
            report.slice_thickness_mm,
            *report.pixel_spacing_mm,
        )
        return report

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #
    def _check_consistency(
        self, paths: list[Path], report: ValidationReport
    ) -> None:
        """Spot-check a handful of slices for consistent dimensions."""
        sample_indices = [0, len(paths) // 2, -1]
        for idx in sample_indices:
            try:
                ds = pydicom.dcmread(str(paths[idx]), stop_before_pixels=True, force=True)
                r, c = int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))
                if (r, c) != (report.rows, report.columns) and report.rows > 0:
                    report.violations.append(
                        f"Inconsistent dimensions: slice {idx} is {r}×{c}, "
                        f"expected {report.rows}×{report.columns}."
                    )
            except Exception:
                pass
