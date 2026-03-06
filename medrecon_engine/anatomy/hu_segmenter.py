"""
medrecon_engine.anatomy.hu_segmenter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Organ-specific HU segmentation — rebuilt clean with all fixes.

Pipeline per organ:

1. **Bones** — HU > 300 → opening → top-20 CC → closing → mesh
2. **Lungs** — HU -950 to -650, body-mask to exclude exterior air,
   top-2 CC, closing
3. **Liver** — HU 40–70 primary / 30–80 fallback, ROI-restricted
   thresholding (right abdomen), 4-tier strategy adaptive to patient
4. **Kidneys** — HU 20–45, z-band 30–70 %, size-filtered CC (50K–800K)

Hierarchical subtraction: bones > lungs > liver > kidneys.

Usage::

    from medrecon_engine.anatomy.hu_segmenter import segment_all
    masks = segment_all(ct_image)
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger
from medrecon_engine.config.hu_ranges import HU_RANGES, STRUCTURE_ORDER

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def segment_all(ct_image: sitk.Image) -> dict[str, sitk.Image]:
    """Segment bones, lungs, liver, kidneys with hierarchical subtraction.

    Returns dict[str, sitk.Image] — only structures with >0 voxels.
    """
    t_start = time.perf_counter()
    _log.info("HU Segmenter: organ-specific pipeline (%d structures)",
              len(STRUCTURE_ORDER))

    _SEGMENTERS: dict[str, Callable] = {
        "bones":   _segment_bones,
        "lungs":   _segment_lungs,
        "liver":   _segment_liver,
        "kidneys": _segment_kidneys,
    }

    masks: dict[str, sitk.Image] = {}
    combined = None  # accumulates all higher-priority masks

    for name in STRUCTURE_ORDER:
        t0 = time.perf_counter()
        spec = HU_RANGES[name]

        raw = _SEGMENTERS[name](ct_image)

        # Remove overlap with higher-priority structures
        if combined is not None:
            raw = sitk.And(raw, sitk.Not(combined))

        nz = int(np.count_nonzero(sitk.GetArrayViewFromImage(raw)))
        dt = time.perf_counter() - t0

        if nz == 0:
            _log.info("    %s: 0 voxels  [%.0f – %.0f HU]  (%.1f s) — skipped",
                      name, spec.hu_low, spec.hu_high, dt)
            continue

        masks[name] = raw
        combined = raw if combined is None else sitk.Or(combined, raw)

        _log.info("    %s: %d voxels  [%.0f – %.0f HU]  (%.1f s)",
                  name, nz, spec.hu_low, spec.hu_high, dt)

    _log.info("HU Segmenter complete: %d masks in %.1f s",
              len(masks), time.perf_counter() - t_start)
    return masks


# ═══════════════════════════════════════════════════════════════════════════
# Organ-specific segmenters
# ═══════════════════════════════════════════════════════════════════════════

def _segment_bones(ct: sitk.Image) -> sitk.Image:
    """Bones: HU > 300 → opening → top-20 CC → closing.

    300 HU excludes intestinal contrast (200–300 HU).
    Opening breaks thin tissue bridges before CC analysis.
    Closing fills internal trabecular pores.
    """
    spec = HU_RANGES["bones"]
    mask = sitk.BinaryThreshold(ct,
                                lowerThreshold=spec.hu_low,
                                upperThreshold=spec.hu_high,
                                insideValue=1, outsideValue=0)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Break thin bridges (bone <-> intestine connections)
    mask = sitk.BinaryMorphologicalOpening(mask, [1, 1, 1], sitk.sitkBall, 0, 1)

    # Keep 20 largest components (spine, pelvis, ribs, femurs, etc.)
    mask = _keep_top_n(mask, 20)

    # Fill trabecular pores
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2], sitk.sitkBall, 1)

    return mask


def _segment_lungs(ct: sitk.Image) -> sitk.Image:
    """Lungs: body mask → threshold -950 to -650 → top-2 CC → closing.

    Body mask (HU > -400) excludes exterior background air that
    otherwise dominates the [-950, -650] range.
    """
    spec = HU_RANGES["lungs"]

    # ── Build body mask to exclude exterior air ──────────────────────
    body = sitk.BinaryThreshold(ct, lowerThreshold=-400, upperThreshold=3000,
                                insideValue=1, outsideValue=0)
    body = sitk.Cast(body, sitk.sitkUInt8)
    body = sitk.BinaryMorphologicalClosing(body, [15, 15, 15], sitk.sitkBall, 1)
    body = sitk.BinaryFillhole(body)

    # ── Lung air inside body only ────────────────────────────────────
    lung_air = sitk.BinaryThreshold(ct,
                                    lowerThreshold=spec.hu_low,
                                    upperThreshold=spec.hu_high,
                                    insideValue=1, outsideValue=0)
    lung_air = sitk.Cast(lung_air, sitk.sitkUInt8)
    mask = sitk.And(lung_air, body)

    # Keep left + right lung
    mask = _keep_top_n(mask, 2)

    # Fill airways
    mask = sitk.BinaryMorphologicalClosing(mask, [4, 4, 4], sitk.sitkBall, 1)

    return mask


def _segment_liver(ct: sitk.Image) -> sitk.Image:
    """Liver: 4-tier ROI-restricted HU thresholding — no region growing.

    Region growing on soft tissue always leaks (entire abdomen is
    connected at liver-range HU values). Instead we use spatial ROI
    restriction + morphological refinement.

    4-tier strategy (prefer opening for controlled size, widen before
    removing opening):
    1. [40, 70] + opening  → L067 wins here (opening prevents oversizing)
    2. [30, 80] + opening  → L143 wins here (wider range, size controlled)
    3. [40, 70] no opening → L096 wins here (opening fragments its liver)
    4. [30, 80] no opening → last resort

    Each tier in right-55% abdomen, z 20–75%.
    Accept threshold: ≥ 400K voxels.
    """
    spec = HU_RANGES["liver"]

    # ── Tier 1: [40, 70] + opening ───────────────────────────────────
    mask = _liver_roi_threshold(ct, spec.hu_low, spec.hu_high, use_opening=True)
    count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
    _log.info("    liver: tier-1 [%.0f, %.0f]+opening → %d voxels",
              spec.hu_low, spec.hu_high, count)

    if count < 400_000:
        # ── Tier 2: [30, 80] + opening ──────────────────────────────
        mask2 = _liver_roi_threshold(ct, 30, 80, use_opening=True)
        count2 = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask2)))
        _log.info("    liver: tier-2 [30, 80]+opening → %d voxels", count2)

        if count2 >= 400_000:
            mask, count = mask2, count2
        else:
            # ── Tier 3: [40, 70] no opening ─────────────────────────
            mask3 = _liver_roi_threshold(ct, spec.hu_low, spec.hu_high, use_opening=False)
            count3 = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask3)))
            _log.info("    liver: tier-3 [%.0f, %.0f] no-opening → %d voxels",
                      spec.hu_low, spec.hu_high, count3)

            if count3 >= 400_000:
                mask, count = mask3, count3
            else:
                # ── Tier 4: [30, 80] no opening ─────────────────────
                mask4 = _liver_roi_threshold(ct, 30, 80, use_opening=False)
                count4 = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask4)))
                _log.info("    liver: tier-4 [30, 80] no-opening → %d voxels", count4)
                mask, count = mask4, count4

    # Close internal vessel gaps
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3], sitk.sitkBall, 1)
    mask = _keep_top_n(mask, 1)

    return mask


def _liver_roi_threshold(
    ct: sitk.Image, hu_low: float, hu_high: float, use_opening: bool = True,
) -> sitk.Image:
    """Threshold + restrict to right abdomen ROI + optional opening + largest CC.

    ROI: z 20-75%, right 55% (x >= 45% of width).
    No anterior/posterior restriction — liver wraps around.
    """
    mask = sitk.BinaryThreshold(ct, lowerThreshold=hu_low, upperThreshold=hu_high,
                                insideValue=1, outsideValue=0)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    zs, ys, xs = arr.shape
    arr[:int(zs * 0.20), :, :] = 0       # above liver
    arr[int(zs * 0.75):, :, :] = 0       # below liver
    arr[:, :, :int(xs * 0.45)] = 0        # left side (keep right 55%)

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    if use_opening:
        # Gentle opening to break thin bridges without fragmenting liver
        mask = sitk.BinaryMorphologicalOpening(mask, [1, 1, 1], sitk.sitkBall, 0, 1)

    # Keep only the largest blob (should be liver)
    mask = _keep_top_n(mask, 1)

    return mask


def _segment_kidneys(ct: sitk.Image) -> sitk.Image:
    """Kidneys: HU 20-45, z-band 30-70 %, size-filtered CC.

    Strategy:
    1. Threshold [20, 45] HU, restrict to z-band 30-70%
    2. Gentle opening [1,1,1] to break thin fascial bridges
    3. Size-filtered CC: keep components 50K-800K voxels (20-320 mL),
       max 2 (left + right kidney)
    4. If nothing qualifies, try without opening
    5. If still nothing, return empty mask (better than returning
       massive muscle blobs that scored poorly in analysis)
    """
    spec = HU_RANGES["kidneys"]

    mask = sitk.BinaryThreshold(ct,
                                lowerThreshold=spec.hu_low,
                                upperThreshold=spec.hu_high,
                                insideValue=1, outsideValue=0)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # z-band: kidneys at 30-70% of axial range
    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    zs = arr.shape[0]
    arr[:int(zs * 0.30), :, :] = 0
    arr[int(zs * 0.70):, :, :] = 0

    mask_zbanded = sitk.GetImageFromArray(arr)
    mask_zbanded.CopyInformation(ct)
    mask_zbanded = sitk.Cast(mask_zbanded, sitk.sitkUInt8)

    # Try with gentle opening first
    opened = sitk.BinaryMorphologicalOpening(
        mask_zbanded, [1, 1, 1], sitk.sitkBall, 0, 1)
    result = _keep_by_size(opened, min_voxels=50_000, max_voxels=800_000, max_count=2)
    count = int(np.count_nonzero(sitk.GetArrayViewFromImage(result)))

    if count == 0:
        # Retry without opening — keeps larger connected regions
        _log.info("    kidneys: opening + size-filter gave 0, retrying without opening")
        result = _keep_by_size(
            mask_zbanded, min_voxels=50_000, max_voxels=800_000, max_count=2)
        count = int(np.count_nonzero(sitk.GetArrayViewFromImage(result)))

    if count == 0:
        _log.info("    kidneys: no kidney-sized components found, returning empty")
        empty = sitk.Image(ct.GetSize(), sitk.sitkUInt8)
        empty.CopyInformation(ct)
        return empty

    # Fill internal gaps
    mask = sitk.BinaryMorphologicalClosing(result, [2, 2, 2], sitk.sitkBall, 1)
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _keep_top_n(mask: sitk.Image, n: int) -> sitk.Image:
    """Keep only the *n* largest connected components."""
    cc = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.Cast(
        sitk.BinaryThreshold(relabeled, 1, n, insideValue=1, outsideValue=0),
        sitk.sitkUInt8,
    )


def _keep_by_size(
    mask: sitk.Image,
    min_voxels: int,
    max_voxels: int,
    max_count: int = 2,
) -> sitk.Image:
    """Keep connected components whose size is within [min, max] voxels.

    Returns up to *max_count* qualifying components, largest first.
    This is critical for kidneys where top-N CC picks muscle blobs
    instead of the smaller kidney-shaped components.
    """
    cc = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    arr = sitk.GetArrayFromImage(relabeled)

    labels = np.unique(arr)
    labels = labels[labels > 0]  # skip background

    kept = 0
    result = np.zeros_like(arr, dtype=np.uint8)

    for label in labels:
        count = int(np.count_nonzero(arr == label))
        if count < min_voxels:
            # Labels are sorted by size — all remaining are smaller
            break
        if count <= max_voxels:
            result[arr == label] = 1
            kept += 1
            _log.info("    size-filter: label %d = %d voxels — kept", label, count)
            if kept >= max_count:
                break
        else:
            _log.info("    size-filter: label %d = %d voxels — too large, skipped",
                      label, count)

    if kept == 0:
        _log.info("    size-filter: no components in [%d, %d] range",
                  min_voxels, max_voxels)
        # Return empty mask — let caller decide what to do
        out = sitk.GetImageFromArray(result)
        out.CopyInformation(mask)
        return sitk.Cast(out, sitk.sitkUInt8)

    out = sitk.GetImageFromArray(result)
    out.CopyInformation(mask)
    return sitk.Cast(out, sitk.sitkUInt8)
