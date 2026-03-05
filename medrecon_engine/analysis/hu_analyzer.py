"""
medrecon_engine.analysis.hu_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute per-organ Hounsfield Unit statistics by overlaying
TotalSegmentator label masks on the original CT volume.

This is the **calibration step** in the hybrid pipeline —
AI masks tell us *where* each organ is, and from those regions
we derive scan-specific HU statistics that feed into adaptive
threshold-based segmentation.

Typical usage::

    ct  = sitk.ReadImage("scan.nii.gz")
    stats = analyze_all_labels(ct, Path("labels/"))
    # → {"liver": {"mean": 58.2, "std": 7.9, ...}, ...}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)

# ── Tissue groups ─────────────────────────────────────────────────────────
# We don't need all 117 labels — only the *key tissues* whose HU ranges
# really matter for downstream segmentation.

TISSUE_GROUPS: dict[str, list[str]] = {
    "bone": [
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
        "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
        "vertebrae_S1",
        "hip_left", "hip_right", "sacrum",
        "femur_left", "femur_right",
        "rib_left_5", "rib_left_6", "rib_right_5", "rib_right_6",
        "sternum", "clavicula_left", "clavicula_right",
        "scapula_left", "scapula_right",
    ],
    "liver": ["liver"],
    "kidney": ["kidney_left", "kidney_right"],
    "spleen": ["spleen"],
    "lung": ["lung_upper_lobe_left", "lung_lower_lobe_left",
             "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    "heart": ["heart"],
    "aorta": ["aorta"],
    "muscle": [
        "gluteus_maximus_left", "gluteus_maximus_right",
        "iliopsoas_left", "iliopsoas_right",
        "autochthon_left", "autochthon_right",
    ],
    "fat": [
        "subcutaneous_fat",     # may not exist in every scan
    ],
}


def calculate_hu_stats(
    ct_array: np.ndarray,
    mask_array: np.ndarray,
) -> dict[str, float] | None:
    """Compute HU statistics for voxels inside a binary mask.

    Parameters
    ----------
    ct_array : np.ndarray
        Full CT volume as numpy (z, y, x), dtype float or int16.
    mask_array : np.ndarray
        Binary mask (0/1) — same shape as *ct_array*.

    Returns
    -------
    dict or None
        ``{"min", "max", "mean", "std", "median", "p5", "p95", "count"}``
        or *None* if the mask is empty.
    """
    values = ct_array[mask_array > 0]
    if values.size == 0:
        return None

    return {
        "min":    float(np.min(values)),
        "max":    float(np.max(values)),
        "mean":   float(np.mean(values)),
        "std":    float(np.std(values)),
        "median": float(np.median(values)),
        "p5":     float(np.percentile(values, 5)),
        "p95":    float(np.percentile(values, 95)),
        "count":  int(values.size),
    }


def analyze_single_label(
    ct_image: sitk.Image,
    mask_path: Path,
    *,
    ct_array: np.ndarray | None = None,
) -> dict[str, float] | None:
    """Read one NIfTI mask and compute its HU stats against the CT.

    Parameters
    ----------
    ct_image : sitk.Image
        Original CT volume.
    mask_path : Path
        Path to a single ``*.nii.gz`` label mask.
    ct_array : np.ndarray | None
        Pre-computed ``sitk.GetArrayFromImage(ct_image)`` to avoid
        repeated conversions.

    Returns
    -------
    dict or None
    """
    if not mask_path.exists():
        return None

    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)

    if ct_array is None:
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

    # Binarise multi-label masks
    if mask_arr.max() > 1:
        mask_arr = (mask_arr > 0).astype(np.uint8)

    return calculate_hu_stats(ct_array, mask_arr)


def analyze_tissue_group(
    ct_image: sitk.Image,
    label_dir: Path,
    group_labels: list[str],
    *,
    ct_array: np.ndarray | None = None,
) -> dict[str, float] | None:
    """Aggregate HU stats across multiple labels for one tissue group.

    E.g. for "bone" we pool all vertebrae, pelvis, femur, ribs, etc.
    to get a single robust bone HU profile.

    Parameters
    ----------
    ct_image : sitk.Image
        Original CT volume.
    label_dir : Path
        Directory containing ``*.nii.gz`` label files.
    group_labels : list[str]
        Label stems to include (e.g. ``["vertebrae_L1", "femur_left"]``).
    ct_array : np.ndarray | None
        Pre-computed CT array.

    Returns
    -------
    dict or None
        Aggregated stats, or *None* if no valid labels found.
    """
    if ct_array is None:
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

    all_values: list[np.ndarray] = []

    for label_stem in group_labels:
        mask_path = label_dir / f"{label_stem}.nii.gz"
        if not mask_path.exists():
            continue

        mask_img = sitk.ReadImage(str(mask_path))
        mask_arr = sitk.GetArrayFromImage(mask_img)
        if mask_arr.max() > 1:
            mask_arr = (mask_arr > 0).astype(np.uint8)

        values = ct_array[mask_arr > 0]
        if values.size > 0:
            all_values.append(values)

    if not all_values:
        return None

    pooled = np.concatenate(all_values)
    return {
        "min":    float(np.min(pooled)),
        "max":    float(np.max(pooled)),
        "mean":   float(np.mean(pooled)),
        "std":    float(np.std(pooled)),
        "median": float(np.median(pooled)),
        "p5":     float(np.percentile(pooled, 5)),
        "p95":    float(np.percentile(pooled, 95)),
        "count":  int(pooled.size),
    }


def analyze_all_labels(
    ct_image: sitk.Image,
    label_dir: str | Path,
) -> dict[str, dict[str, float]]:
    """Compute HU profiles for all tissue groups defined in TISSUE_GROUPS.

    Parameters
    ----------
    ct_image : sitk.Image
        Original CT volume.
    label_dir : str | Path
        Directory with TotalSegmentator label masks.

    Returns
    -------
    dict[str, dict]
        ``{"bone": {"mean": 780, ...}, "liver": {...}, ...}``
    """
    label_dir = Path(label_dir)
    t0 = time.perf_counter()
    _log.info("HU Analyzer: computing per-tissue statistics …")

    # Extract CT array once
    ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

    profiles: dict[str, dict[str, float]] = {}

    for tissue_name, labels in TISSUE_GROUPS.items():
        stats = analyze_tissue_group(
            ct_image, label_dir, labels, ct_array=ct_array,
        )
        if stats is not None:
            profiles[tissue_name] = stats
            _log.info(
                "  %-10s  mean=%7.1f  std=%6.1f  [%7.1f … %7.1f]  n=%d",
                tissue_name,
                stats["mean"],
                stats["std"],
                stats["p5"],
                stats["p95"],
                stats["count"],
            )
        else:
            _log.info("  %-10s  (no labels found)", tissue_name)

    elapsed = time.perf_counter() - t0
    _log.info("HU Analyzer complete: %d tissue profiles in %.1f s", len(profiles), elapsed)
    return profiles


def save_hu_profile(
    profiles: dict[str, dict[str, float]],
    output_path: str | Path,
    *,
    scan_id: str = "",
) -> Path:
    """Write the HU profile to a JSON file.

    Parameters
    ----------
    profiles : dict
        Output of :func:`analyze_all_labels`.
    output_path : str | Path
        Where to write the JSON.
    scan_id : str
        Optional scan identifier to embed in the file.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"scan_id": scan_id}
    payload["profiles"] = profiles

    # Also emit compact range summaries for easy consumption
    ranges: dict[str, list[float]] = {}
    for tissue, stats in profiles.items():
        ranges[f"{tissue}_range"] = [
            round(stats["p5"], 1),
            round(stats["p95"], 1),
        ]
    payload["ranges"] = ranges

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    _log.info("HU profile saved → %s", output_path)
    return output_path
