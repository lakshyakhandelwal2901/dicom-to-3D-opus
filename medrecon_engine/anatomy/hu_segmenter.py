"""
medrecon_engine.anatomy.hu_segmenter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Organ-specific HU segmentation — rebuilt clean with all fixes.

Pipeline per organ:

1. **Bones** — HU > 300 → opening → top-20 CC → closing → mesh
2. **Lungs** — HU -950 to -650, body-mask to exclude exterior air,
   top-2 CC, closing
3. **Liver** — HU 40–70, seeded region growing from right-abdomen,
   fallback to ROI-restricted wider threshold
4. **Kidneys** — HU 20–45, z-band restriction (T12–L3), top-2 CC

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
    """Liver: seeded region growing [40-70 HU] from right-abdomen seed.

    Falls back to a wider ROI-restricted threshold [30-80 HU] if
    region growing produces an unreasonable volume.
    """
    spec = HU_RANGES["liver"]

    seed = _find_liver_seed(ct, spec.hu_low, spec.hu_high)

    if seed is not None:
        grown = sitk.ConnectedThreshold(
            ct, seedList=[seed],
            lower=spec.hu_low, upper=spec.hu_high,
            replaceValue=1,
        )
        grown = sitk.Cast(grown, sitk.sitkUInt8)
        count = int(np.count_nonzero(sitk.GetArrayViewFromImage(grown)))
        _log.info("    liver: region growing from %s -> %d voxels", seed, count)

        if 50_000 < count < 5_000_000:
            mask = grown
        else:
            _log.info("    liver: region growing out of range, using ROI fallback")
            mask = _liver_roi_fallback(ct)
    else:
        _log.info("    liver: no seed found, using ROI fallback")
        mask = _liver_roi_fallback(ct)

    # Close internal vessel gaps
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3], sitk.sitkBall, 1)
    mask = _keep_top_n(mask, 1)

    return mask


def _liver_roi_fallback(ct: sitk.Image) -> sitk.Image:
    """Fallback: wider HU [30,80] restricted to right-abdomen ROI."""
    mask = sitk.BinaryThreshold(ct, lowerThreshold=30, upperThreshold=80,
                                insideValue=1, outsideValue=0)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    zs, ys, xs = arr.shape
    arr[:int(zs * 0.25), :, :] = 0       # above liver
    arr[int(zs * 0.75):, :, :] = 0       # below liver
    arr[:, :, :xs // 2] = 0              # left side

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Light opening to break thin bridges
    mask = sitk.BinaryMorphologicalOpening(mask, [1, 1, 1], sitk.sitkBall, 0, 1)
    mask = _keep_top_n(mask, 1)
    return mask


def _segment_kidneys(ct: sitk.Image) -> sitk.Image:
    """Kidneys: HU 20-45, z-band 30-70%, top-2 CC, closing.

    Z-band restriction isolates the T12-L3 vertebrae region
    where kidneys reside.
    """
    spec = HU_RANGES["kidneys"]

    mask = sitk.BinaryThreshold(ct,
                                lowerThreshold=spec.hu_low,
                                upperThreshold=spec.hu_high,
                                insideValue=1, outsideValue=0)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Z-band restriction: kidneys at 30-70% of axial range
    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    zs = arr.shape[0]
    arr[:int(zs * 0.30), :, :] = 0
    arr[int(zs * 0.70):, :, :] = 0

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Left + right kidney
    mask = _keep_top_n(mask, 2)

    # Fill internal gaps
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3], sitk.sitkBall, 1)

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


def _find_liver_seed(
    ct: sitk.Image, hu_low: float, hu_high: float,
) -> tuple[int, int, int] | None:
    """Find a seed voxel in the right-abdomen for liver region growing.

    Searches the right half of the volume, middle 50% of z,
    and returns a voxel near the centroid of qualifying voxels
    whose HU is within [hu_low, hu_high].

    Returns (x, y, z) in SimpleITK index order, or None.
    """
    arr = sitk.GetArrayViewFromImage(ct)  # (z, y, x)
    zs, ys, xs = arr.shape

    z0, z1 = int(zs * 0.25), int(zs * 0.75)
    x_mid = xs // 2

    roi = arr[z0:z1, :, x_mid:]
    qualifying = (roi >= hu_low) & (roi <= hu_high)
    nz = np.nonzero(qualifying)

    if len(nz[0]) < 100:
        _log.info("    liver: only %d qualifying voxels in ROI", len(nz[0]))
        return None

    # Centroid of qualifying region
    cz = int(np.mean(nz[0])) + z0
    cy = int(np.mean(nz[1]))
    cx = int(np.mean(nz[2])) + x_mid

    # Verify seed is in range
    if hu_low <= float(arr[cz, cy, cx]) <= hu_high:
        _log.info("    liver: seed at (%d,%d,%d) val=%.1f HU",
                  cx, cy, cz, float(arr[cz, cy, cx]))
        return (int(cx), int(cy), int(cz))

    # Search nearby 10-voxel radius for a valid seed
    _log.info("    liver: centroid val=%.1f HU out of range, searching nearby",
              float(arr[cz, cy, cx]))
    best_d, best = float("inf"), None
    for dz in range(-10, 11):
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                sz, sy, sx = cz + dz, cy + dy, cx + dx
                if 0 <= sz < zs and 0 <= sy < ys and 0 <= sx < xs:
                    v = float(arr[sz, sy, sx])
                    if hu_low <= v <= hu_high:
                        d = dz * dz + dy * dy + dx * dx
                        if d < best_d:
                            best_d, best = d, (int(sx), int(sy), int(sz))

    if best:
        _log.info("    liver: nearby seed at %s", best)
    return best
