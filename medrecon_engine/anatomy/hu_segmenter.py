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
    """Bones: threshold > 300 HU, keep top 20 CC, closing.

    Pipeline:
      1. Trabecular (300–900 HU) | Cortical (>900 HU) combined mask
      2. Morphological opening [1,1,1] — break thin bridges to intestine
      3. Keep top-20 connected components (all major skeletal structures)
      4. Binary closing [2,2,2] — fills internal trabecular pores
    """
    spec = HU_RANGES["bones"]
    mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=spec.hu_low,
        upperThreshold=spec.hu_high,
        insideValue=1, outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Break thin tissue bridges between bone and intestinal contrast
    mask = sitk.BinaryMorphologicalOpening(mask, [1, 1, 1], sitk.sitkBall, 0, 1)

    # Keep top 20 largest connected components
    mask = _keep_top_n_components(mask, n=20)

    # Fill internal trabecular pores and small gaps
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2], sitk.sitkBall, 1)

    return mask


def _segment_lungs(ct_image: sitk.Image) -> sitk.Image:
    """Lungs: threshold -950 to -650 HU, exclude exterior air, keep 2 CC.

    Outside-body air also falls in [-950, -650] so we must exclude it.
    Pipeline:
      1. Build a body mask (HU > -400, fill holes, largest CC)
      2. Threshold [-950, -650] AND body interior
      3. Keep 2 largest CC (left + right lung)
      4. Morphological closing [4,4,4] to fill airways
    """
    spec = HU_RANGES["lungs"]

    # Step 1: body mask — everything above -400 HU is "not air"
    body = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=-400,
        upperThreshold=3000,
        insideValue=1, outsideValue=0,
    )
    body = sitk.Cast(body, sitk.sitkUInt8)
    # Close gaps in the body silhouette
    body = sitk.BinaryMorphologicalClosing(body, [15, 15, 15], sitk.sitkBall, 1)
    # Fill interior holes so lungs are inside the body
    body = sitk.BinaryFillhole(body)

    # Step 2: lung-air threshold intersected with body interior
    lung_air = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=spec.hu_low,
        upperThreshold=spec.hu_high,
        insideValue=1, outsideValue=0,
    )
    lung_air = sitk.Cast(lung_air, sitk.sitkUInt8)
    mask = sitk.And(lung_air, body)

    # Step 3: keep only the two largest components (left + right lung)
    mask = _keep_top_n_components(mask, n=2)

    # Step 4: fill airways / holes with morphological closing
    mask = sitk.BinaryMorphologicalClosing(mask, [4, 4, 4], sitk.sitkBall, 1)

    return mask


def _segment_liver(ct_image: sitk.Image) -> sitk.Image:
    """Liver: seeded region growing 40–70 HU from right-abdomen seed.

    Pipeline:
      1. Find seed point in right-abdomen (where liver always is)
      2. ConnectedThreshold region growing [40, 70]
      3. If region growing is reasonable, use it; otherwise ROI fallback
      4. Morphological closing [3,3,3] to fill vessels
      5. Keep largest CC only
    """
    spec = HU_RANGES["liver"]

    # Step 1: find seed in right abdomen
    seed = _find_liver_seed(ct_image, spec.hu_low, spec.hu_high)

    if seed is not None:
        # Step 2: seeded region growing
        grown = sitk.ConnectedThreshold(
            ct_image,
            seedList=[seed],
            lower=spec.hu_low,
            upper=spec.hu_high,
            replaceValue=1,
        )
        grown = sitk.Cast(grown, sitk.sitkUInt8)
        grown_count = int(np.count_nonzero(sitk.GetArrayViewFromImage(grown)))
        _log.info("    liver: region growing from %s → %d voxels", seed, grown_count)

        # If region growing is reasonable (<5M voxels), use it
        if 50_000 < grown_count < 5_000_000:
            mask = grown
        else:
            _log.info("    liver: region growing out of range, using ROI fallback")
            mask = _liver_roi_fallback(ct_image)
    else:
        _log.info("    liver: no seed found, using ROI fallback")
        mask = _liver_roi_fallback(ct_image)

    # Step 3: morphological closing to fill internal vessels
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3], sitk.sitkBall, 1)

    # Step 4: keep only the single largest blob
    mask = _keep_top_n_components(mask, n=1)

    return mask


def _liver_roi_fallback(ct_image: sitk.Image) -> sitk.Image:
    """Fallback liver segmentation: wider HU range [30,80] in right-abdomen ROI."""
    # Use wider HU range for fallback since diffusion blurs boundaries
    mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=30,
        upperThreshold=80,
        insideValue=1, outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    zs, ys, xs = arr.shape
    z0, z1 = int(zs * 0.25), int(zs * 0.75)
    x_mid = xs // 2
    arr[:z0, :, :] = 0
    arr[z1:, :, :] = 0
    arr[:, :, :x_mid] = 0

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct_image)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Light opening to break thin bridges (less aggressive than [3,3,3])
    mask = sitk.BinaryMorphologicalOpening(mask, [1, 1, 1], sitk.sitkBall, 0, 1)
    mask = _keep_top_n_components(mask, n=1)
    return mask


def _segment_kidneys(ct_image: sitk.Image) -> sitk.Image:
    """Kidneys: threshold 20–45 HU, restrict to T12–L3, keep top 2.

    Pipeline:
      1. HU threshold [20, 45]
      2. Restrict to z=30–70% (T12–L3 vertebrae region)
      3. Keep top-2 CC (left + right kidney)
      4. Closing [3,3,3]
    """
    spec = HU_RANGES["kidneys"]

    # Step 1: HU threshold
    mask = sitk.BinaryThreshold(
        ct_image,
        lowerThreshold=spec.hu_low,
        upperThreshold=spec.hu_high,
        insideValue=1, outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Step 2: restrict to T12–L3 vertical band
    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)
    zs, ys, xs = arr.shape

    z_start = int(zs * 0.30)
    z_end   = int(zs * 0.70)

    # Zero outside kidney z-band
    arr[:z_start, :, :] = 0
    arr[z_end:,   :, :] = 0

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ct_image)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Step 3: keep the two largest blobs (left + right kidney)
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
