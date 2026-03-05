"""
medrecon_engine.anatomy.hu_segmenter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Organ-specific HU segmentation with structure-aware cleanup.

Each organ uses different thresholding + filtering logic because
HU ranges overlap between soft tissues.  The pipeline is:

1. **Bones** — threshold > 250 HU, keep top-20 connected components
2. **Lungs** — threshold -950 to -650 HU, keep top-2 CC, morphological closing
3. **Liver** — threshold 40–70 HU, region-growing from right-abdomen seed,
   morphological closing
4. **Kidneys** — threshold 20–45 HU, restrict to mid-volume vertical band,
   keep top-2 symmetric blobs

Hierarchical subtraction prevents overlap: bones are subtracted from
lungs, both are subtracted from liver, etc.

Usage::

    from medrecon_engine.anatomy.hu_segmenter import segment_all
    masks = segment_all(ct_image)          # dict[str, sitk.Image]
"""

from __future__ import annotations

import time

import numpy as np
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger
from medrecon_engine.config.hu_ranges import HU_RANGES, STRUCTURE_ORDER

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def segment_all(ct_image: sitk.Image) -> dict[str, sitk.Image]:
    """Segment bones, lungs, liver, kidneys with hierarchical overlap removal.

    Parameters
    ----------
    ct_image : sitk.Image
        CT volume in Hounsfield Units (optionally pre-smoothed with
        anisotropic diffusion).

    Returns
    -------
    dict[str, sitk.Image]
        Mapping structure name → cleaned binary mask (uint8, 0/1).
        Only structures with > 0 foreground voxels are included.
    """
    t_start = time.perf_counter()
    _log.info("HU Segmenter: organ-specific pipeline (%d structures)",
              len(STRUCTURE_ORDER))

    # Dispatch table for organ-specific segmenters
    _SEGMENTERS = {
        "bones":   _segment_bones,
        "lungs":   _segment_lungs,
        "liver":   _segment_liver,
        "kidneys": _segment_kidneys,
    }

    masks: dict[str, sitk.Image] = {}
    combined_higher = None  # accumulates all higher-priority masks

    for structure in STRUCTURE_ORDER:
        t0 = time.perf_counter()
        spec = HU_RANGES[structure]
        segmenter = _SEGMENTERS[structure]

        # Organ-specific segmentation
        raw_mask = segmenter(ct_image)

        # Subtract all previously segmented (higher-priority) masks
        if combined_higher is not None:
            raw_mask = sitk.And(raw_mask, sitk.Not(combined_higher))

        # Count voxels
        arr = sitk.GetArrayViewFromImage(raw_mask)
        nz = int(np.count_nonzero(arr))

        if nz == 0:
            _log.info("    %s: 0 voxels  [%.0f – %.0f HU]  (%.1f s) — skipped",
                      structure, spec.hu_low, spec.hu_high,
                      time.perf_counter() - t0)
            continue

        masks[structure] = raw_mask

        # Accumulate for overlap removal
        if combined_higher is None:
            combined_higher = sitk.Cast(raw_mask, sitk.sitkUInt8)
        else:
            combined_higher = sitk.Or(combined_higher, raw_mask)

        _log.info("    %s: %d voxels  [%.0f – %.0f HU]  (%.1f s)",
                  structure, nz, spec.hu_low, spec.hu_high,
                  time.perf_counter() - t0)

    elapsed = time.perf_counter() - t_start
    _log.info("HU Segmenter complete: %d masks in %.1f s", len(masks), elapsed)
    return masks


# ═══════════════════════════════════════════════════════════════════════════
# Organ-specific segmenters
# ═══════════════════════════════════════════════════════════════════════════

def _segment_bones(ct_image: sitk.Image) -> sitk.Image:
    """Bones: threshold > 250 HU, keep top 20 connected components.

    Captures both trabecular (250–900) and cortical (900–3000) bone
    in a single pass.  The top-20 CC filter keeps major skeletal
    structures (spine, ribs, pelvis, femurs, etc.) and removes noise.
    """
    spec = HU_RANGES["bones"]
    mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=spec.hu_low,
        upperThreshold=spec.hu_high,
        insideValue=1, outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Keep top 20 largest connected components
    mask = _keep_top_n_components(mask, n=20)

    return mask


def _segment_lungs(ct_image: sitk.Image) -> sitk.Image:
    """Lungs: threshold -950 to -650 HU, keep 2 largest CC, close holes.

    Lungs are very reliable due to extreme HU values.  The airways
    create internal holes which are filled with morphological closing.
    """
    spec = HU_RANGES["lungs"]
    mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=spec.hu_low,
        upperThreshold=spec.hu_high,
        insideValue=1, outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Keep only the two lungs (left + right)
    mask = _keep_top_n_components(mask, n=2)

    # Fill airways / holes with morphological closing
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3], sitk.sitkBall, 1)

    return mask


def _segment_liver(ct_image: sitk.Image) -> sitk.Image:
    """Liver: threshold 30–80 HU, restricted to right-abdomen ROI.

    Region-growing leaks into surrounding soft tissue because the
    liver shares HU values with muscle / intestines.  Instead we:
    1. Threshold [30, 80] on the whole volume
    2. Zero out everything outside the right-abdomen quadrant
    3. Morphological opening to break thin tissue bridges
    4. Keep only the single largest blob (= liver)
    5. Morphological closing to fill internal vessels / gaps
    """
    spec = HU_RANGES["liver"]

    # Step 1: global HU threshold
    mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=spec.hu_low,
        upperThreshold=spec.hu_high,
        insideValue=1, outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Step 2: restrict to right-abdomen ROI
    # Liver sits in the right half (x > midpoint), middle 60% of z
    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    zs, ys, xs = arr.shape

    z0 = int(zs * 0.20)
    z1 = int(zs * 0.80)
    x_mid = xs // 2

    # Zero out left half and top/bottom z
    arr[:z0, :, :] = 0
    arr[z1:, :, :] = 0
    arr[:, :, :x_mid] = 0

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct_image)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Step 3: morphological opening to break thin bridges to other tissues
    mask = sitk.BinaryMorphologicalOpening(mask, [3, 3, 3], sitk.sitkBall, 0, 1)

    # Step 4: keep the single largest connected component (the liver)
    mask = _keep_top_n_components(mask, n=1)

    # Step 5: morphological closing to fill internal vessels / gaps
    mask = sitk.BinaryMorphologicalClosing(mask, [5, 5, 5], sitk.sitkBall, 1)

    return mask


def _segment_kidneys(ct_image: sitk.Image) -> sitk.Image:
    """Kidneys: threshold 20–45 HU, restrict to mid-volume, keep top 2.

    Kidneys sit roughly in the middle 40% of the axial (z) range
    and posterior half of the volume.  We crop the search region
    to reduce false positives from overlapping soft tissue.
    """
    spec = HU_RANGES["kidneys"]

    # Step 1: HU threshold on full volume
    mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=spec.hu_low,
        upperThreshold=spec.hu_high,
        insideValue=1, outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Step 2: restrict to mid-volume vertical band (T12–L3 region)
    # Kidneys are roughly in the middle 40% of z-axis
    size = ct_image.GetSize()   # (x, y, z)
    z_total = size[2]
    z_start = int(z_total * 0.30)
    z_end   = int(z_total * 0.70)

    # Zero out voxels outside the kidney vertical band
    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    arr[:z_start, :, :] = 0
    arr[z_end:,   :, :] = 0

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct_image)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Step 3: keep the two largest symmetric blobs
    mask = _keep_top_n_components(mask, n=2)

    # Step 4: morphological closing
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3], sitk.sitkBall, 1)

    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _keep_top_n_components(mask: sitk.Image, *, n: int) -> sitk.Image:
    """Keep only the *n* largest connected components."""
    cc = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    cleaned = sitk.BinaryThreshold(
        relabeled,
        lowerThreshold=1,
        upperThreshold=n,
        insideValue=1,
        outsideValue=0,
    )
    return sitk.Cast(cleaned, sitk.sitkUInt8)


def _find_liver_seed(
    ct_image: sitk.Image,
    hu_low: float,
    hu_high: float,
) -> tuple[int, int, int] | None:
    """Find a seed point for liver region growing.

    The liver is the largest organ in the right-anterior quadrant
    of the abdomen.  We search the right half, middle 50% of z,
    and pick a voxel near the centroid of the qualifying region
    whose HU value is actually within [hu_low, hu_high].

    Returns (x_idx, y_idx, z_idx) in SimpleITK index order, or None.
    """
    arr = sitk.GetArrayViewFromImage(ct_image)  # (z, y, x)
    zs, ys, xs = arr.shape

    # Search region: right half (x > midpoint), middle 50% of z
    z0, z1 = int(zs * 0.25), int(zs * 0.75)
    x_mid = xs // 2

    roi = arr[z0:z1, :, x_mid:]

    # Find voxels in liver HU range within the ROI
    liver_voxels = (roi >= hu_low) & (roi <= hu_high)
    nz = np.nonzero(liver_voxels)

    if len(nz[0]) < 100:
        _log.info("    liver: only %d qualifying voxels in ROI — no seed",
                  len(nz[0]))
        return None

    # Compute centroid of the qualifying region
    cz_roi = int(np.mean(nz[0]))
    cy_roi = int(np.mean(nz[1]))
    cx_roi = int(np.mean(nz[2]))

    # Map back to full-volume indices
    cz = cz_roi + z0
    cy = cy_roi
    cx = cx_roi + x_mid

    # Verify the seed voxel itself is within the HU range
    val = float(arr[cz, cy, cx])
    if hu_low <= val <= hu_high:
        _log.info("    liver: seed at (%d,%d,%d) val=%.1f HU", cx, cy, cz, val)
        return (int(cx), int(cy), int(cz))

    # Centroid value not in range — search nearby (5×5×5 cube) for a
    # qualifying voxel closest to the centroid
    _log.info("    liver: centroid (%d,%d,%d) val=%.1f HU out of range, searching nearby",
              cx, cy, cz, val)
    best_dist = float("inf")
    best_seed = None
    radius = 10
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sz, sy, sx = cz + dz, cy + dy, cx + dx
                if 0 <= sz < zs and 0 <= sy < ys and 0 <= sx < xs:
                    v = float(arr[sz, sy, sx])
                    if hu_low <= v <= hu_high:
                        d = dz * dz + dy * dy + dx * dx
                        if d < best_dist:
                            best_dist = d
                            best_seed = (int(sx), int(sy), int(sz))

    if best_seed is not None:
        _log.info("    liver: nearby seed at %s", best_seed)
    else:
        _log.info("    liver: no qualifying voxel near centroid")

    return best_seed
