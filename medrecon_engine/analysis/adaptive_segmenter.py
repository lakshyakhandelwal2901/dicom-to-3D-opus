"""
medrecon_engine.analysis.adaptive_segmenter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classic HU + morphology segmentation driven by **adaptive thresholds**
derived from TotalSegmentator calibration.

Instead of hard-coded ``bone > 200 HU``, this module uses per-scan
thresholds computed from actual tissue statistics.  The AI tells us
*where* each organ is; from that we learn *what HU values* to expect;
then we apply classic image-processing to get high-detail masks.

Pipeline per tissue::

    CT volume
      ↓
    HU threshold → binary mask
      ↓
    morphological close (fill small holes)
      ↓
    connected-component filtering (remove small islands)
      ↓
    optional: AND with AI mask (restrict to known region)
      ↓
    output binary mask (SimpleITK Image)

The resulting masks feed into the existing mesh pipeline
(gradient-guided FlyingEdges → Taubin smooth → OBJ).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)

# ── Per-tissue morphology parameters ─────────────────────────────────────

_MORPH_PARAMS: dict[str, dict] = {
    "bone": {
        "close_radius": 2,             # close small gaps in cortex
        "min_component_voxels": 500,    # drop tiny bone fragments
        "use_ai_mask": False,           # bone thresholding is reliable
    },
    "liver": {
        "close_radius": 3,
        "min_component_voxels": 5000,
        "use_ai_mask": True,            # restrict to AI liver region
    },
    "kidney": {
        "close_radius": 2,
        "min_component_voxels": 2000,
        "use_ai_mask": True,
    },
    "spleen": {
        "close_radius": 2,
        "min_component_voxels": 2000,
        "use_ai_mask": True,
    },
    "lung": {
        "close_radius": 3,
        "min_component_voxels": 10000,
        "use_ai_mask": False,           # lung HU is very distinctive
    },
    "heart": {
        "close_radius": 3,
        "min_component_voxels": 5000,
        "use_ai_mask": True,
    },
    "aorta": {
        "close_radius": 2,
        "min_component_voxels": 1000,
        "use_ai_mask": True,            # aorta HU overlaps muscle
    },
    "muscle": {
        "close_radius": 2,
        "min_component_voxels": 3000,
        "use_ai_mask": True,
    },
}


def segment_tissue(
    ct_image: sitk.Image,
    tissue_name: str,
    hu_low: float,
    hu_high: float,
    ai_mask: sitk.Image | None = None,
) -> sitk.Image | None:
    """Segment a single tissue using adaptive HU thresholding + morphology.

    Parameters
    ----------
    ct_image : sitk.Image
        Original CT volume (float32).
    tissue_name : str
        Tissue key (e.g. ``"bone"``, ``"liver"``).
    hu_low, hu_high : float
        Adaptive HU range from :mod:`threshold_generator`.
    ai_mask : sitk.Image | None
        Combined AI mask for this tissue (union of all TotalSegmentator
        labels in the group).  Used to restrict the classic mask to the
        known anatomical region when ``use_ai_mask=True``.

    Returns
    -------
    sitk.Image | None
        Binary mask (0/1, sitkUInt8), or *None* if segmentation is empty.
    """
    params = _MORPH_PARAMS.get(tissue_name, {
        "close_radius": 2,
        "min_component_voxels": 500,
        "use_ai_mask": True,
    })

    t0 = time.perf_counter()

    # 1 — HU thresholding
    ct_f = sitk.Cast(ct_image, sitk.sitkFloat32)
    mask = sitk.BinaryThreshold(
        ct_f,
        lowerThreshold=hu_low,
        upperThreshold=hu_high,
        insideValue=1,
        outsideValue=0,
    )

    # 2 — Restrict to AI region (if available and enabled)
    if params["use_ai_mask"] and ai_mask is not None:
        # Dilate the AI mask slightly to allow the classic threshold
        # to capture edge voxels the AI might have missed
        dilated = sitk.BinaryDilate(
            sitk.Cast(ai_mask, sitk.sitkUInt8),
            kernelRadius=[3, 3, 3],
        )
        mask = sitk.And(mask, dilated)

    # 3 — Morphological closing (fill holes)
    radius = params["close_radius"]
    if radius > 0:
        mask = sitk.BinaryMorphologicalClosing(
            sitk.Cast(mask, sitk.sitkUInt8),
            kernelRadius=[radius, radius, radius],
        )

    # 4 — Connected-component filtering (remove small islands)
    min_voxels = params["min_component_voxels"]
    if min_voxels > 0:
        mask = _remove_small_components(mask, min_voxels)

    # Check if anything remains
    arr = sitk.GetArrayFromImage(mask)
    nz = int(np.count_nonzero(arr))
    if nz == 0:
        _log.info("    %s: empty after segmentation", tissue_name)
        return None

    elapsed = time.perf_counter() - t0
    _log.info(
        "    %s: %d voxels  [%.1f – %.1f HU]  (%.2f s)",
        tissue_name,
        nz,
        hu_low,
        hu_high,
        elapsed,
    )

    return sitk.Cast(mask, sitk.sitkUInt8)


def build_ai_group_mask(
    label_dir: Path,
    label_stems: list[str],
) -> sitk.Image | None:
    """Load and OR-combine multiple TotalSegmentator labels into one mask.

    Parameters
    ----------
    label_dir : Path
        Directory with ``*.nii.gz`` label files.
    label_stems : list[str]
        Label stems to combine (e.g. ``["kidney_left", "kidney_right"]``).

    Returns
    -------
    sitk.Image | None
        Combined binary mask, or *None* if no labels found.
    """
    combined: sitk.Image | None = None

    for stem in label_stems:
        path = label_dir / f"{stem}.nii.gz"
        if not path.exists():
            continue

        img = sitk.ReadImage(str(path))
        binary = sitk.BinaryThreshold(img, 1, 10000, 1, 0)

        if combined is None:
            combined = binary
        else:
            combined = sitk.Or(combined, binary)

    return combined


def segment_all_tissues(
    ct_image: sitk.Image,
    thresholds: dict[str, tuple[float, float]],
    label_dir: str | Path | None = None,
) -> dict[str, sitk.Image]:
    """Run adaptive segmentation for all tissues with known thresholds.

    Parameters
    ----------
    ct_image : sitk.Image
        Original CT volume.
    thresholds : dict[str, tuple[float, float]]
        Adaptive thresholds from :mod:`threshold_generator`.
    label_dir : str | Path | None
        TotalSegmentator labels directory.  If provided, AI masks are
        used to constrain segmentation for tissues that benefit from it.

    Returns
    -------
    dict[str, sitk.Image]
        ``{"bone": <mask>, "liver": <mask>, ...}``
    """
    from medrecon_engine.analysis.hu_analyzer import TISSUE_GROUPS

    label_dir = Path(label_dir) if label_dir else None
    t0 = time.perf_counter()

    _log.info("Adaptive Segmenter: processing %d tissues …", len(thresholds))

    results: dict[str, sitk.Image] = {}

    for tissue_name, (hu_low, hu_high) in thresholds.items():
        # Build AI group mask if labels are available
        ai_mask: sitk.Image | None = None
        if label_dir is not None and tissue_name in TISSUE_GROUPS:
            ai_mask = build_ai_group_mask(
                label_dir,
                TISSUE_GROUPS[tissue_name],
            )

        mask = segment_tissue(
            ct_image,
            tissue_name,
            hu_low,
            hu_high,
            ai_mask=ai_mask,
        )

        if mask is not None:
            results[tissue_name] = mask

    elapsed = time.perf_counter() - t0
    _log.info(
        "Adaptive Segmenter complete: %d tissue masks in %.1f s",
        len(results),
        elapsed,
    )

    return results


# ── Internal helpers ──────────────────────────────────────────────────────

def _remove_small_components(
    mask: sitk.Image,
    min_voxels: int,
) -> sitk.Image:
    """Remove connected components smaller than *min_voxels*."""
    cc = sitk.ConnectedComponent(sitk.Cast(mask, sitk.sitkUInt8))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    # Build a set of labels to keep
    keep_labels = set()
    for label in stats.GetLabels():
        if stats.GetNumberOfPixels(label) >= min_voxels:
            keep_labels.add(label)

    if not keep_labels:
        return sitk.Image(mask.GetSize(), sitk.sitkUInt8)

    # Create output mask
    cc_arr = sitk.GetArrayFromImage(cc)
    out_arr = np.zeros_like(cc_arr, dtype=np.uint8)
    for label in keep_labels:
        out_arr[cc_arr == label] = 1

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(mask)
    return out
