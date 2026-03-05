"""
MedRecon Engine v1 — main entry point
======================================
Universal CT → Surgical 3D STL Engine

This module implements the **complete 14-step pipeline** and exposes both a
Python API (``run_case``) and a Click-based CLI with batch-mode support.

14-Step Workflow
----------------
 1. Scan dataset directory for DICOM series
 2. Validate DICOM headers (precision-mode hard fail)
 3. Load DICOM volume → SimpleITK Image
 4. Convert to calibrated Hounsfield Units
 5. Adaptive HU estimation for the target anatomy
 6. Segment anatomy mask
 7. (internal) Connected-component filtering
 8. Resample to isotropic grid
 9. VTK marching-cubes mesh generation
10. Mesh post-processing (smooth → decimate → fill holes)
11. Topology validation
12. Quality / confidence scoring
13. Export STL
14. Write audit record

Usage (CLI)
-----------
::

    # Single case
    python -m medrecon_engine.main --dicom ./data/patient_01 --anatomy bone --output ./output

    # Batch (all anatomies)
    python -m medrecon_engine.main --dicom ./data/patient_01 --anatomy bone lung brain --output ./output

    # Batch directory (multiple patients)
    python -m medrecon_engine.main --batch-dir ./data --anatomy bone --output ./output
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

import click
import numpy as np
import SimpleITK as sitk
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from medrecon_engine.audit.logger import (
    AuditRecord,
    get_logger,
    write_audit_record,
)
from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.core.dataset_scanner import DatasetScanner
from medrecon_engine.core.dicom_validator import DicomValidator
from medrecon_engine.core.hu_converter import HUConverter
from medrecon_engine.core.preprocessing import Preprocessor
from medrecon_engine.core.resampler import Resampler
from medrecon_engine.core.volume_loader import VolumeLoader
from medrecon_engine.export.stl_writer import STLWriter
from medrecon_engine.hu_model.hu_estimator import HUEstimator
from medrecon_engine.mesh.mesh_postprocess import MeshPostProcessor
from medrecon_engine.mesh.mesh_validator import MeshValidator
from medrecon_engine.mesh.vtk_generator import generate_mesh
from medrecon_engine.quality.confidence_score import ConfidenceScorer
from medrecon_engine.quality.surface_metrics import SurfaceMetricsComputer
from medrecon_engine.anatomy.registry import get_segmenter, list_anatomies

_log = get_logger(__name__)
console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# Core pipeline function
# ═══════════════════════════════════════════════════════════════════════════
def run_case(
    dicom_path: str | Path,
    anatomy: str,
    output_dir: str | Path,
    *,
    config: Optional[PrecisionConfig] = None,
) -> AuditRecord:
    """Execute the full 14-step pipeline for a single (patient, anatomy) pair.

    Parameters
    ----------
    dicom_path : str | Path
        Path to a directory containing DICOM files (may have sub-dirs).
    anatomy : str
        Target anatomy key (``bone``, ``lung``, ``brain``, …).
    output_dir : str | Path
        Root output directory.  STL + audit files are written here.
    config : PrecisionConfig | None
        Override default precision configuration.

    Returns
    -------
    AuditRecord
        Frozen record summarising the pipeline run.
    """
    cfg = config or PrecisionConfig()
    dicom_path = Path(dicom_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    _log.info("═" * 60)
    _log.info("MedRecon Engine v1  —  anatomy=%s", anatomy)
    _log.info("DICOM source : %s", dicom_path)
    _log.info("Output dir   : %s", output_dir)
    _log.info("═" * 60)

    try:
        # ── Step 1: Scan dataset ──────────────────────────────────────
        _log.info("[1/14] Scanning dataset …")
        scanner = DatasetScanner()
        best_series = scanner.select_best_ct(dicom_path)
        series_uid = best_series.series_uid
        files = best_series.file_paths
        _log.info("  Selected series %s  (%d files)", series_uid[:16], len(files))

        # ── Step 2: Validate DICOM headers ────────────────────────────
        _log.info("[2/14] Validating DICOM compliance …")
        validator = DicomValidator(cfg)
        validator.validate(files)
        _log.info("  Validation PASSED")

        # ── Step 3: Load volume ───────────────────────────────────────
        _log.info("[3/14] Loading volume …")
        loader = VolumeLoader()
        volume = loader.load(files)
        spacing = volume.GetSpacing()
        size = volume.GetSize()
        _log.info("  Volume %s  spacing %.2f×%.2f×%.2f mm", size, *spacing)

        # ── Step 4: HU conversion ────────────────────────────────────
        _log.info("[4/14] Converting to calibrated HU …")
        hu_conv = HUConverter()
        volume = hu_conv.convert(volume)

        # ── Step 5: Adaptive HU estimation ────────────────────────────
        _log.info("[5/14] Adaptive HU estimation for '%s' …", anatomy)
        estimator = HUEstimator()
        vol_arr = sitk.GetArrayFromImage(volume).astype(np.float64)
        hu_est = estimator.estimate(vol_arr, anatomy)
        hu_range = (hu_est.adaptive_min, hu_est.adaptive_max)
        _log.info("  Adaptive HU range: [%.1f, %.1f]  peak=%.1f", hu_range[0], hu_range[1], hu_est.detected_peak)

        # ── Step 6: Preprocessing (smooth) ────────────────────────────
        # NOTE: We do NOT crop_to_body here — cropping destroys the
        # physical origin/direction, causing VTK geometry corruption.
        # Instead we only smooth in-place, preserving full SITK metadata.
        _log.info("[6/14] Preprocessing (smooth) …")
        preproc = Preprocessor(cfg)
        vol_arr = sitk.GetArrayFromImage(volume).astype(np.float64)
        vol_arr = preproc.smooth(vol_arr, spacing=volume.GetSpacing())
        smoothed = sitk.GetImageFromArray(vol_arr)
        smoothed.CopyInformation(volume)   # preserve spacing + origin + direction
        volume = smoothed

        # ── Step 7: Segment anatomy ───────────────────────────────────
        _log.info("[7/14] Segmenting '%s' …", anatomy)
        segmenter = get_segmenter(anatomy, config=cfg)

        mask_sitk = segmenter.segment(volume)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        voxel_count = int(np.sum(mask_arr > 0))
        voxel_vol_mm3 = float(np.prod(volume.GetSpacing()))
        vol_cm3 = (voxel_count * voxel_vol_mm3) / 1000.0
        _log.info("  Segmented voxels: %d  (≈ %.1f cm³)", voxel_count, vol_cm3)

        if voxel_count == 0:
            raise ValueError(
                f"Segmentation produced empty mask for anatomy='{anatomy}'. "
                "Check HU range or input data."
            )

        # ── Step 8: Resample to isotropic grid ───────────────────────
        _log.info("[8/14] Resampling to isotropic grid …")
        resampler = Resampler(cfg)
        mask_sitk = resampler.resample(mask_sitk)
        volume = resampler.resample(volume)

        # ── Step 9: VTK marching cubes ───────────────────────────────
        _log.info("[9/14] VTK marching cubes …")
        mesh = generate_mesh(mask_sitk)
        _log.info(
            "  Raw mesh: %d pts, %d faces",
            mesh.GetNumberOfPoints(),
            mesh.GetNumberOfCells(),
        )

        # ── Step 10: Post-process (smooth, decimate, fill) ───────────
        _log.info("[10/14] Post-processing mesh …")
        postproc = MeshPostProcessor(cfg)
        mesh = postproc.process(mesh)
        _log.info(
            "  Final mesh: %d pts, %d faces",
            mesh.GetNumberOfPoints(),
            mesh.GetNumberOfCells(),
        )

        # ── Step 11: Topology validation ─────────────────────────────
        _log.info("[11/14] Validating mesh topology …")
        mesh_val = MeshValidator(cfg)
        val_report = mesh_val.validate(mesh)
        _log.info("  Manifold: %s  |  Degenerate faces: %d", val_report.is_manifold, val_report.num_degenerate_faces)

        # ── Step 12: Quality / confidence scoring ────────────────────
        _log.info("[12/14] Computing quality metrics …")
        metrics_comp = SurfaceMetricsComputer()
        metrics = metrics_comp.compute(mesh)

        scorer = ConfidenceScorer(cfg)
        score_report = scorer.compute(
            validation=val_report,
            metrics=metrics,
            hu_estimation=hu_est,
            original_spacing=tuple(float(s) for s in spacing),
        )
        _log.info(
            "  Confidence: %.3f  Grade: %s",
            score_report.total,
            score_report.grade,
        )

        # ── Step 13: Export STL ──────────────────────────────────────
        _log.info("[13/14] Exporting STL …")
        stl_name = f"{anatomy}_{series_uid[:8]}.stl"
        stl_path = output_dir / stl_name
        writer = STLWriter(cfg)
        export_report = writer.write(
            mesh, stl_path, solid_name=f"medrecon_{anatomy}"
        )
        _log.info("  STL → %s  (%.1f KB)", stl_path.name, export_report.file_size_bytes / 1024)

        # ── Step 14: Audit record ────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        _log.info("[14/14] Writing audit record …")
        record = AuditRecord(
            dicom_path=str(dicom_path),
            anatomy=anatomy,
            num_slices=len(files),
            voxel_spacing_mm=tuple(round(s, 4) for s in spacing),
            output_stl_path=str(stl_path),
            mesh_vertices=mesh.GetNumberOfPoints(),
            mesh_faces=mesh.GetNumberOfCells(),
            volume_cm3=round(vol_cm3, 2),
            confidence_score=round(score_report.total, 4),
            confidence_grade=score_report.grade,
            topology_valid=val_report.passed,
            manifold=val_report.is_manifold,
            hu_adaptive_min=round(hu_range[0], 2),
            hu_adaptive_max=round(hu_range[1], 2),
            elapsed_seconds=round(elapsed, 3),
            success=True,
        )
        audit_path = write_audit_record(record, audit_dir=output_dir)
        _log.info("  Audit → %s", audit_path.name)
        _log.info("═" * 60)
        _log.info("DONE  %s  —  %.1f s  —  Grade: %s", anatomy.upper(), elapsed, score_report.grade)
        _log.info("STL  : %s", stl_path)
        _log.info("═" * 60)

        return record

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        _log.error("Pipeline FAILED for anatomy='%s': %s", anatomy, exc, exc_info=True)

        record = AuditRecord(
            dicom_path=str(dicom_path),
            anatomy=anatomy,
            elapsed_seconds=round(elapsed, 3),
            success=False,
            error_message=str(exc),
        )
        write_audit_record(record, audit_dir=output_dir)
        return record


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════
def run_batch(
    dicom_path: str | Path,
    anatomies: Sequence[str],
    output_dir: str | Path,
    *,
    config: Optional[PrecisionConfig] = None,
) -> List[AuditRecord]:
    """Run the pipeline for multiple anatomies on the same dataset."""
    records = []
    for anat in anatomies:
        records.append(run_case(dicom_path, anat, output_dir, config=config))
    return records


def run_batch_dir(
    batch_dir: str | Path,
    anatomies: Sequence[str],
    output_root: str | Path,
    *,
    config: Optional[PrecisionConfig] = None,
) -> List[AuditRecord]:
    """Run the pipeline across multiple patient directories.

    Each immediate sub-directory of *batch_dir* is treated as a patient
    DICOM folder.  Outputs are written to ``output_root/<patient_name>/``.
    """
    batch_dir = Path(batch_dir)
    output_root = Path(output_root)
    records = []

    patient_dirs = sorted(
        d for d in batch_dir.iterdir() if d.is_dir()
    )
    _log.info("Batch mode: %d patient dirs, %d anatomies", len(patient_dirs), len(anatomies))

    for pdir in patient_dirs:
        out = output_root / pdir.name
        for anat in anatomies:
            records.append(run_case(pdir, anat, out, config=config))

    return records


# ═══════════════════════════════════════════════════════════════════════════
# Rich summary table
# ═══════════════════════════════════════════════════════════════════════════
def _print_summary(records: List[AuditRecord]) -> None:
    """Pretty-print a summary of pipeline results."""
    table = Table(title="MedRecon Engine — Pipeline Summary", show_lines=True)
    table.add_column("Run ID", style="dim")
    table.add_column("Anatomy", style="bold cyan")
    table.add_column("Status")
    table.add_column("Confidence", justify="right")
    table.add_column("Grade")
    table.add_column("Vertices", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Volume (cm³)", justify="right")
    table.add_column("Time (s)", justify="right")

    for r in records:
        status = "[green]✓ OK[/green]" if r.success else f"[red]✗ {r.error_message[:30]}[/red]"
        grade_colour = {
            "SURGICAL": "green",
            "USABLE": "yellow",
            "REJECT": "red",
        }.get(r.confidence_grade, "white")

        table.add_row(
            r.run_id,
            r.anatomy,
            status,
            f"{r.confidence_score:.3f}" if r.success else "—",
            f"[{grade_colour}]{r.confidence_grade}[/{grade_colour}]" if r.confidence_grade else "—",
            f"{r.mesh_vertices:,}" if r.success else "—",
            f"{r.mesh_faces:,}" if r.success else "—",
            f"{r.volume_cm3:.1f}" if r.success else "—",
            f"{r.elapsed_seconds:.1f}",
        )

    console.print()
    console.print(table)
    console.print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
@click.command("medrecon")
@click.option(
    "--dicom", "-d",
    type=click.Path(exists=True, file_okay=False),
    help="Path to DICOM directory (single patient).",
)
@click.option(
    "--batch-dir", "-b",
    type=click.Path(exists=True, file_okay=False),
    help="Path to batch directory containing multiple patient folders.",
)
@click.option(
    "--anatomy", "-a",
    multiple=True,
    default=["bone"],
    help="Anatomy targets (can repeat: -a bone -a lung). Use 'all' for every registered anatomy.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./output",
    show_default=True,
    help="Output root directory for STL + audit files.",
)
@click.option(
    "--precision-mm",
    type=float,
    default=None,
    help="Override strict precision threshold (mm).",
)
@click.option(
    "--no-smooth",
    is_flag=True,
    default=False,
    help="Disable mesh smoothing.",
)
@click.option(
    "--ascii-stl",
    is_flag=True,
    default=False,
    help="Export ASCII STL instead of binary.",
)
def cli(
    dicom: Optional[str],
    batch_dir: Optional[str],
    anatomy: tuple,
    output: str,
    precision_mm: Optional[float],
    no_smooth: bool,
    ascii_stl: bool,
) -> None:
    """MedRecon Engine v1 — Universal CT → Surgical 3D STL Engine."""
    console.print(
        Panel.fit(
            "[bold cyan]MedRecon Engine v1[/bold cyan]\n"
            "Universal CT → Surgical 3D STL Engine\n"
            "[dim]Deterministic · Audit-ready · On-prem[/dim]",
            border_style="cyan",
        )
    )

    if not dicom and not batch_dir:
        console.print("[red]Error:[/red] Provide --dicom or --batch-dir.")
        sys.exit(1)

    # ── Resolve anatomy list ─────────────────────────────────────────
    if "all" in anatomy:
        anatomies = list_anatomies()
    else:
        anatomies = list(anatomy)

    console.print(f"  Anatomies : {', '.join(anatomies)}")
    console.print(f"  Output    : {output}")

    # ── Build config overrides ────────────────────────────────────────
    overrides = {}
    if precision_mm is not None:
        overrides["strict_precision_mm"] = precision_mm
    if no_smooth:
        overrides["mesh_smooth_iterations"] = 0
    if ascii_stl:
        overrides["stl_binary"] = False

    cfg = PrecisionConfig(**overrides) if overrides else PrecisionConfig()

    # ── Run ───────────────────────────────────────────────────────────
    if batch_dir:
        records = run_batch_dir(batch_dir, anatomies, output, config=cfg)
    else:
        records = run_batch(dicom, anatomies, output, config=cfg)

    _print_summary(records)

    # ── Exit code: non-zero if any case failed ────────────────────────
    failures = sum(1 for r in records if not r.success)
    if failures:
        console.print(f"[red]{failures} case(s) failed.[/red]")
        sys.exit(1)
    else:
        console.print("[green]All cases completed successfully.[/green]")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cli()
