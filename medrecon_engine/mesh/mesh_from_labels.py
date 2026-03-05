"""
medrecon_engine.mesh.mesh_from_labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convert TotalSegmentator label masks (NIfTI) into high-quality VTK meshes
using **gradient-guided surface extraction**.

Instead of extracting iso-surfaces from binary masks (which produce voxel
stair-step artefacts), this module uses the original CT intensity gradients
restricted to each organ's segmented region.  This preserves sub-voxel
anatomical detail that binary masks destroy.

Pipeline per organ::

    CT volume ─┐
               ├─► crop to organ bbox
    Mask ──────┘
               |
     Gaussian smooth (variance=1.2)
               |
     GradientMagnitude
               |
     Multiply by organ mask   <-- gradient only inside the organ
               |
     FlyingEdges3D on gradient field
               |
     FillHoles -> Taubin smooth -> QuadricDecimation -> Normals

Each organ is processed on a **tight bounding-box crop** to keep
memory usage manageable (a full 512×512×560 float32 volume is ~587 MB;
cropping to the organ region reduces this to kilobytes–megabytes).

If no CT volume is supplied, falls back to the classic binary-mask
iso-surface extraction.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk

from medrecon_engine.audit.logger import get_logger
from medrecon_engine.mesh.vtk_generator import (
    generate_mesh,
    improve_mesh,
)

_log = get_logger(__name__)

# Padding (in voxels) around the organ bounding box when cropping
_BBOX_PAD = 5


def generate_meshes_from_labels(
    label_dir: str | Path,
    ct_image: sitk.Image | None = None,
    *,
    gauss_variance: float = 1.2,
    gradient_iso: float = 0.1,
) -> dict[str, vtk.vtkPolyData]:
    """Convert every NIfTI label mask into a VTK mesh.

    When *ct_image* is provided the function uses **gradient-guided
    surface extraction** for sub-voxel accuracy.  Otherwise it falls
    back to binary-mask FlyingEdges.

    Parameters
    ----------
    label_dir : str | Path
        Directory containing ``*.nii.gz`` masks (one per organ).
    ct_image : sitk.Image | None
        Original CT volume (full resolution, HU).  Pass this to enable
        gradient-guided extraction.
    gauss_variance : float
        Gaussian smoothing variance for gradient computation (mm²).
    gradient_iso : float
        Iso-surface value on the masked gradient field.

    Returns
    -------
    dict[str, vtkPolyData]
        Mapping from label name to post-processed VTK mesh.
    """
    label_dir = Path(label_dir)
    use_gradient = ct_image is not None

    nifti_files = sorted(label_dir.glob("*.nii.gz"))
    _log.info("Found %d label masks in %s", len(nifti_files), label_dir)
    _log.info(
        "  Mode: %s",
        "gradient-guided (sub-voxel)" if use_gradient else "binary mask (classic)",
    )

    # Cast CT once (avoids repeated per-organ casts)
    ct_f: sitk.Image | None = None
    if use_gradient:
        ct_f = sitk.Cast(ct_image, sitk.sitkFloat32)

    results: dict[str, vtk.vtkPolyData] = {}
    skipped = 0

    for nifti_path in nifti_files:
        label_name = nifti_path.name.replace(".nii.gz", "")
        t0 = time.perf_counter()

        _log.info("  Processing: %s", label_name)

        # Read mask
        label_img = sitk.ReadImage(str(nifti_path))
        arr = sitk.GetArrayFromImage(label_img)

        # Skip empty masks
        nonzero = int(np.count_nonzero(arr))
        if nonzero == 0:
            _log.info("    empty mask, skipping")
            skipped += 1
            continue

        # Binarise
        if arr.max() > 1:
            label_img = sitk.BinaryThreshold(
                label_img,
                lowerThreshold=1,
                upperThreshold=int(arr.max()),
                insideValue=1,
                outsideValue=0,
            )

        _log.info("    voxels=%d", nonzero)

        # ── Generate mesh ──────────────────────────────────────────
        if use_gradient:
            mesh = _gradient_mesh_for_label(
                ct_f, label_img, arr, gauss_variance, gradient_iso,
            )
        else:
            mesh = generate_mesh(label_img)

        if mesh is None or mesh.GetNumberOfCells() == 0:
            _log.info("    mesh empty after iso-surface, skipping")
            skipped += 1
            continue

        # Post-process: FillHoles -> Taubin smooth -> Decimate -> Normals
        raw_faces = mesh.GetNumberOfCells()
        mesh = improve_mesh(
            mesh,
            smooth_iterations=30,     # more iterations for gradient meshes
            smooth_passband=0.08,      # tighter passband for medical quality
        )
        _log.info(
            "    mesh: %d pts, %d faces  (raw %d)",
            mesh.GetNumberOfPoints(),
            mesh.GetNumberOfCells(),
            raw_faces,
        )

        results[label_name] = mesh

        elapsed = time.perf_counter() - t0
        _log.info("    done in %.1f s", elapsed)

    _log.info(
        "Label -> Mesh complete: %d meshes, %d skipped",
        len(results),
        skipped,
    )
    return results


# ── Internal helpers ──────────────────────────────────────────────────────

def _organ_bbox(arr: np.ndarray) -> tuple[list[int], list[int]]:
    """Compute a padded bounding-box (start_index, crop_size) in (x, y, z).

    *arr* is the **numpy** mask array in (z, y, x) order.
    Returns SimpleITK-convention (x, y, z) index & size.
    """
    nz = np.nonzero(arr)
    z0, z1 = int(nz[0].min()), int(nz[0].max())
    y0, y1 = int(nz[1].min()), int(nz[1].max())
    x0, x1 = int(nz[2].min()), int(nz[2].max())

    # Pad and clamp
    z0, y0, x0 = max(0, z0 - _BBOX_PAD), max(0, y0 - _BBOX_PAD), max(0, x0 - _BBOX_PAD)
    z1 = min(arr.shape[0] - 1, z1 + _BBOX_PAD)
    y1 = min(arr.shape[1] - 1, y1 + _BBOX_PAD)
    x1 = min(arr.shape[2] - 1, x1 + _BBOX_PAD)

    start_idx = [x0, y0, z0]               # SimpleITK order
    crop_size = [x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1]
    return start_idx, crop_size


def _gradient_mesh_for_label(
    ct_f: sitk.Image,
    mask_image: sitk.Image,
    mask_arr: np.ndarray,
    gauss_variance: float,
    iso_value: float,
) -> vtk.vtkPolyData | None:
    """Extract a gradient-guided iso-surface for a single label.

    Crops both the CT and mask to the organ bounding-box first,
    then computes the gradient on the tiny crop.  This keeps memory
    usage per organ at kilobytes–megabytes instead of ~600 MB.
    """
    from medrecon_engine.mesh.vtk_generator import _sitk_to_vtk

    # 1 — Crop to organ bbox
    start_idx, crop_size = _organ_bbox(mask_arr)
    ct_crop = sitk.RegionOfInterest(ct_f, crop_size, start_idx)
    mask_crop = sitk.RegionOfInterest(mask_image, crop_size, start_idx)

    # 2 — Smooth + gradient on the small crop
    smoothed = sitk.DiscreteGaussian(ct_crop, variance=gauss_variance)
    gradient = sitk.GradientMagnitude(smoothed)

    # 3 — Restrict gradient to the organ region
    mask_crop_f = sitk.Cast(mask_crop, sitk.sitkFloat32)
    refined = gradient * mask_crop_f

    # 4 — Convert to VTK and extract iso-surface
    vtk_img = _sitk_to_vtk(refined, scalar_type="float")

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(vtk_img)
    fe.SetValue(0, iso_value)
    fe.ComputeNormalsOn()
    fe.Update()

    return fe.GetOutput()
