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
4. **Kidneys** — spine-guided tubular-ROI segmentation: track kidney
   centroids per slice relative to vertebral body, create 35 mm tube,
   threshold HU 10–55 inside tube, largest CC, Gaussian smooth
5. **Kidney stones** — HU > 150 in kidney + ureter + bladder corridor

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

        # Remove overlap with higher-priority structures.
        # Exception: kidneys — do NOT subtract liver.  Liver HU (30-80)
        # overlaps kidney HU (20-45) and the liver ROI extends into the
        # right kidney region, destroying the kidney mask.  Physical
        # overlap is impossible (separate organs); we only subtract
        # bones + lungs whose HU ranges don't actually overlap with
        # kidneys anyway (effectively a no-op safety net).
        if name == "kidneys":
            bones_lungs = None
            for prior in ("bones", "lungs"):
                if prior in masks:
                    bones_lungs = (masks[prior] if bones_lungs is None
                                   else sitk.Or(bones_lungs, masks[prior]))
            if bones_lungs is not None:
                raw = sitk.And(raw, sitk.Not(bones_lungs))
        elif combined is not None:
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

    # ── Kidney stones: derived from kidney region ────────────────────
    if "kidneys" in masks:
        _log.info("  Extracting kidney stones from kidney region …")
        t0 = time.perf_counter()
        stones = _segment_kidney_stones(ct_image, masks["kidneys"])
        nz = int(np.count_nonzero(sitk.GetArrayViewFromImage(stones)))
        dt = time.perf_counter() - t0
        if nz > 0:
            masks["kidney_stones"] = stones
            _log.info("    kidney_stones: %d voxels  [>200 HU in kidney region]  (%.1f s)",
                      nz, dt)
        else:
            _log.info("    kidney_stones: none detected  (%.1f s)", dt)

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
    """Kidneys: spine-guided tubular-ROI segmentation.

    The fundamental challenge: kidney parenchyma (HU 20-45) overlaps
    with psoas / paraspinal muscle (HU 35-60), forming one giant
    connected blob at these HU values.  Region growing and simple
    thresholding always leak into adjacent muscles.

    Solution — **spine-guided tubular ROI**:

    1. Find vertebral body position per axial slice (bone landmark).
    2. Per-slice 2D connected-component tracking to locate kidney
       centroids at the known lateral offset from spine (~60 mm
       lateral, ~10 mm anterior).  Each CC is scored by
       size / (1 + distance_to_expected), best wins.
    3. Smooth centroid trajectory along Z with a running median and
       reject outliers (> 25 mm from smoothed trajectory).
    4. Interpolate trajectory across all slices in the kidney Z-range.
    5. Create a 35 mm-radius tubular ROI following the trajectory.
    6. HU threshold [10, 55] within tube → 3D largest connected
       component = kidney.
    7. Morphological closing + fill-holes + Gaussian smooth for a
       clean anatomical 3D surface.

    This works because the tube is tight enough to exclude psoas and
    paraspinal muscles yet wide enough to capture the full kidney
    cross-section.  The per-slice tracking avoids false CC selection
    by anchoring on the spine landmark.
    """
    import scipy.ndimage as _ndi
    from scipy.interpolate import interp1d as _interp1d
    from scipy.signal import medfilt as _medfilt

    arr_ct = sitk.GetArrayViewFromImage(ct)
    zs, ys, xs = arr_ct.shape
    spacing = ct.GetSpacing()  # (x_sp, y_sp, z_sp)
    x_sp, y_sp, z_sp = spacing

    # ── Z-band ───────────────────────────────────────────────────────
    z_lo = int(zs * 0.25)
    z_hi = int(zs * 0.72)

    def _arr_to_sitk(arr: np.ndarray) -> sitk.Image:
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(ct)
        return img

    # ── Step 1: Spine landmark per axial slice ───────────────────────
    _log.info("    kidneys: locating spine landmarks …")
    spine_x = np.full(z_hi - z_lo, xs // 2, dtype=np.intp)
    spine_y = np.full(z_hi - z_lo, ys // 2, dtype=np.intp)

    for zi, z in enumerate(range(z_lo, z_hi)):
        bone_slice = arr_ct[z] > 250
        if not bone_slice.any():
            continue
        labeled, n_cc = _ndi.label(bone_slice)
        sizes = _ndi.sum(np.ones_like(labeled), labeled, range(1, n_cc + 1))
        best = int(np.argmax(sizes)) + 1
        ys_c, xs_c = np.where(labeled == best)
        spine_x[zi] = int(np.median(xs_c))
        spine_y[zi] = int(np.median(ys_c))

    med_spine_x = int(np.median(spine_x))
    _log.info("    kidneys: spine median X=%d, Y=%d",
              med_spine_x, int(np.median(spine_y)))

    # ── Step 2: Per-slice 2D CC tracking ─────────────────────────────
    TUBE_RADIUS_MM = 35.0
    MIN_CC_PX = 100
    MAX_CC_PX = 6000
    EXPECTED_LAT_MM = 60.0  # lateral offset from spine
    EXPECTED_ANT_MM = 10.0  # anterior offset from spine
    MAX_DIST_MM = 60.0
    MIN_LAT_MM = 20.0       # minimum lateral offset (exclude midline)

    result = np.zeros((zs, ys, xs), dtype=np.uint8)
    found_sides = 0

    for side_name, side_sign in [("left", -1), ("right", 1)]:
        # Collect centroids for this side
        centroids = []  # (zi, cx, cy, z_global)

        for zi, z in enumerate(range(z_lo, z_hi)):
            slc = arr_ct[z]
            sx_c, sy_c = int(spine_x[zi]), int(spine_y[zi])

            # Soft-tissue threshold in posterior half of body
            soft = (slc >= 0) & (slc <= 60)
            y_lo_post = max(0, sy_c - int(70))  # ~55 mm anterior
            y_hi_post = min(ys, sy_c + int(15))  # ~12 mm posterior
            post_mask = np.zeros_like(soft)
            post_mask[y_lo_post:y_hi_post, :] = True
            soft = soft & post_mask

            labeled_2d, n_cc = _ndi.label(soft)
            if n_cc == 0:
                continue
            cc_sizes = _ndi.sum(
                np.ones_like(labeled_2d), labeled_2d, range(1, n_cc + 1))

            expected_x = sx_c + side_sign * int(EXPECTED_LAT_MM / x_sp)
            expected_y = sy_c - int(EXPECTED_ANT_MM / y_sp)

            best_score = -1.0
            best_info = None

            for lbl in range(1, n_cc + 1):
                sz_v = int(cc_sizes[lbl - 1])
                if sz_v < MIN_CC_PX or sz_v > MAX_CC_PX:
                    continue

                ys_i, xs_i = np.where(labeled_2d == lbl)
                cx, cy = float(np.median(xs_i)), float(np.median(ys_i))

                # Must be on correct side of spine
                lat_offset_mm = (cx - sx_c) * side_sign * x_sp
                if lat_offset_mm < MIN_LAT_MM:
                    continue

                dx_mm = (cx - expected_x) * x_sp
                dy_mm = (cy - expected_y) * y_sp
                dist = np.sqrt(dx_mm ** 2 + dy_mm ** 2)
                if dist > MAX_DIST_MM:
                    continue

                score = sz_v / (1.0 + dist)
                if score > best_score:
                    best_score = score
                    best_info = (zi, cx, cy, float(z))

            if best_info is not None:
                centroids.append(best_info)

        if len(centroids) < 10:
            _log.info("    kidneys: %s — only %d tracking points, skipping",
                      side_name, len(centroids))
            continue

        centroids = np.array(centroids)
        cx_all = centroids[:, 1]
        cy_all = centroids[:, 2]
        z_all = centroids[:, 3]

        # ── Step 3: Smooth trajectory + outlier rejection ────────────
        k = min(21, len(cx_all) if len(cx_all) % 2 == 1
                else len(cx_all) - 1)
        k = max(3, k)
        smooth_cx = _medfilt(cx_all, kernel_size=k)
        smooth_cy = _medfilt(cy_all, kernel_size=k)

        dists = np.sqrt(((cx_all - smooth_cx) * x_sp) ** 2
                        + ((cy_all - smooth_cy) * y_sp) ** 2)
        keep = dists < 25.0  # 25 mm tolerance
        good = centroids[keep]

        if len(good) < 10:
            _log.info("    kidneys: %s — only %d points after filtering, skipping",
                      side_name, len(good))
            continue

        _log.info("    kidneys: %s — %d trajectory points (%d kept)",
                  side_name, len(centroids), len(good))

        # Re-smooth after filtering
        k2 = max(3, min(21, len(good) if len(good) % 2 == 1
                         else len(good) - 1))
        smooth_cx2 = _medfilt(good[:, 1], kernel_size=k2)
        smooth_cy2 = _medfilt(good[:, 2], kernel_size=k2)

        # ── Step 4: Interpolate across full Z-range ──────────────────
        f_cx = _interp1d(good[:, 3], smooth_cx2,
                         kind="linear", fill_value="extrapolate")
        f_cy = _interp1d(good[:, 3], smooth_cy2,
                         kind="linear", fill_value="extrapolate")

        z_range_min = int(good[:, 3].min())
        z_range_max = int(good[:, 3].max())

        # ── Step 5: Tubular ROI + HU threshold ──────────────────────
        kidney_tube = np.zeros((zs, ys, xs), dtype=bool)
        Y_grid, X_grid = np.ogrid[:ys, :xs]

        for z in range(z_range_min, z_range_max + 1):
            cx_z = float(f_cx(z))
            cy_z = float(f_cy(z))
            dist_sq = ((X_grid - cx_z) * x_sp) ** 2 \
                    + ((Y_grid - cy_z) * y_sp) ** 2
            circle = dist_sq <= (TUBE_RADIUS_MM ** 2)
            kidney_tube[z] = circle & (arr_ct[z] >= 10) & (arr_ct[z] <= 55)

        # ── Step 6: Largest 3D connected component ───────────────────
        labeled_3d, n_3d = _ndi.label(kidney_tube)
        if n_3d == 0:
            _log.info("    kidneys: %s — tube produced no components", side_name)
            continue

        sizes_3d = _ndi.sum(
            np.ones_like(labeled_3d), labeled_3d, range(1, n_3d + 1))
        best_3d = int(np.argmax(sizes_3d)) + 1
        kidney_side = (labeled_3d == best_3d).astype(np.uint8)

        nz_side = int(np.count_nonzero(kidney_side))
        vol_ml = nz_side * x_sp * y_sp * z_sp / 1000.0

        if nz_side < 20_000:  # < ~10 mL
            _log.info("    kidneys: %s — too small (%d vox, %.0f mL), skipping",
                      side_name, nz_side, vol_ml)
            continue

        result[kidney_side > 0] = 1
        found_sides += 1
        _log.info("    kidneys: %s → %d voxels (%.0f mL)",
                  side_name, nz_side, vol_ml)

    # ── Result ───────────────────────────────────────────────────────
    count = int(np.count_nonzero(result))
    if count == 0:
        _log.info("    kidneys: spine-guided approach found nothing, returning empty")
        empty = sitk.Image(ct.GetSize(), sitk.sitkUInt8)
        empty.CopyInformation(ct)
        return empty

    # ── Step 7: Morphological cleanup for clean surface ──────────────
    kidney_mask = _arr_to_sitk(result)
    kidney_mask = sitk.BinaryMorphologicalClosing(
        kidney_mask, [4, 4, 3], sitk.sitkBall, 1)
    kidney_mask = sitk.BinaryFillhole(kidney_mask)

    # Gaussian smooth → re-threshold for smooth 3D surface
    mask_f = sitk.Cast(kidney_mask, sitk.sitkFloat32)
    mask_f = sitk.DiscreteGaussian(mask_f, variance=4.0)
    kidney_mask = sitk.BinaryThreshold(
        mask_f, lowerThreshold=0.4, upperThreshold=1.0,
        insideValue=1, outsideValue=0,
    )
    kidney_mask = sitk.Cast(kidney_mask, sitk.sitkUInt8)

    return kidney_mask


def _segment_kidney_stones(ct: sitk.Image, kidney_mask: sitk.Image) -> sitk.Image:
    """Extract kidney/urinary stones: high-density calcifications in the
    kidneys, ureters (kidney→bladder tracks), and bladder.

    Stones can be anywhere in the urinary tract:
    - Renal pelvis / calyces (inside kidney)
    - Ureter (narrow tube from kidney down to bladder)
    - Urinary bladder (pelvic floor)

    Strategy:
    1. Create a urinary-tract search corridor:
       - Dilate kidney mask moderately (8 mm) for renal stones
       - Extend a vertical column downward from each kidney to
         the pelvic floor (z < 25%) for ureteral/bladder stones
    2. Threshold CT > 150 HU within that corridor (calcifications).
    3. Size-filter: keep stone-sized components (5 – 8 K voxels),
       up to 30 individual stones.
    """
    # ── Build urinary tract search corridor ──────────────────────────
    # A) Dilated kidney region (renal stones in pelvis/calyces)
    renal_zone = sitk.BinaryDilate(kidney_mask, [10, 10, 6], sitk.sitkBall)

    # B) Ureteral / bladder corridor: vertical column below kidneys
    #    Find the x-range of each kidney, extend z downward
    k_arr = sitk.GetArrayViewFromImage(kidney_mask)
    zs, ys, xs = k_arr.shape
    mid_x = xs // 2

    corridor = np.zeros((zs, ys, xs), dtype=np.uint8)

    # Left kidney → left ureter column
    left_region = k_arr[:, :, :mid_x]
    if np.any(left_region):
        z_idx, y_idx, x_idx = np.where(left_region)
        x_center = int(np.median(x_idx))
        y_center = int(np.median(y_idx))
        z_bottom = int(np.min(z_idx))
        # Column from bottom of kidney down to pelvis, ~3cm wide
        r = max(20, int(0.03 * xs))  # ~3 cm radius in voxels
        y_lo = max(0, y_center - r)
        y_hi = min(ys, y_center + r)
        x_lo = max(0, x_center - r)
        x_hi = min(mid_x, x_center + r)
        corridor[:z_bottom, y_lo:y_hi, x_lo:x_hi] = 1

    # Right kidney → right ureter column
    right_region = k_arr[:, :, mid_x:]
    if np.any(right_region):
        z_idx, y_idx, x_idx = np.where(right_region)
        x_center = int(np.median(x_idx)) + mid_x
        y_center = int(np.median(y_idx))
        z_bottom = int(np.min(z_idx))
        r = max(20, int(0.03 * xs))
        y_lo = max(0, y_center - r)
        y_hi = min(ys, y_center + r)
        x_lo = max(mid_x, x_center - r)
        x_hi = min(xs, x_center + r)
        corridor[:z_bottom, y_lo:y_hi, x_lo:x_hi] = 1

    corridor_img = sitk.GetImageFromArray(corridor)
    corridor_img.CopyInformation(ct)
    corridor_img = sitk.Cast(corridor_img, sitk.sitkUInt8)

    # Combine: renal zone + ureteral/bladder corridor
    search = sitk.Or(sitk.Cast(renal_zone, sitk.sitkUInt8), corridor_img)

    # ── Threshold for calcifications in the search zone ──────────────
    dense = sitk.BinaryThreshold(
        ct, lowerThreshold=150.0, upperThreshold=3000.0,
        insideValue=1, outsideValue=0,
    )
    dense = sitk.Cast(dense, sitk.sitkUInt8)
    stones = sitk.And(dense, search)

    # ── Size-filter: stones are small ────────────────────────────────
    # Typical stone: 2–20 mm → ~8 to ~8600 voxels at <1 mm spacing
    stones = _keep_by_size(stones, min_voxels=5, max_voxels=8_000, max_count=30)

    return stones


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
