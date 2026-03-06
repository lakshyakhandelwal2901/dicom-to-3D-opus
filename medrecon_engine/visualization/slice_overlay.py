"""
medrecon_engine.visualization.slice_overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
2D CT slice overlay visualization with segmentation masks
using matplotlib.

Features
--------
- Axial / coronal / sagittal slice views
- Colour-coded organ overlays on each slice
- Multi-slice montage generation
- Segmentation coverage summary

Usage::

    from medrecon_engine.visualization.slice_overlay import (
        save_slice_overlay,
        save_montage,
    )

    save_slice_overlay(ct_image, masks, output_dir / "overlay.png")
    save_montage(ct_image, masks, output_dir / "montage.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger
from medrecon_engine.visualization.colors import OVERLAY_COLORS

_log = get_logger(__name__)


def _safe_import_plt():
    """Import matplotlib with Agg backend (headless-safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ct_to_display(ct_slice: np.ndarray,
                   window_center: float = 40,
                   window_width: float = 400) -> np.ndarray:
    """Apply CT windowing for display (soft tissue by default)."""
    lo = window_center - window_width / 2
    hi = window_center + window_width / 2
    clipped = np.clip(ct_slice, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def _blend_overlay(
    ct_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    alpha: float = 0.35,
) -> np.ndarray:
    """Blend organ mask colours onto a grayscale CT slice.

    Parameters
    ----------
    ct_rgb : (H, W, 3) uint8 grayscale repeated to RGB
    masks : dict[organ_name, (H, W) bool array]
    alpha : overlay opacity

    Returns
    -------
    (H, W, 3) uint8 blended image
    """
    out = ct_rgb.astype(np.float32)

    for organ, m in masks.items():
        if m.sum() == 0:
            continue
        color = OVERLAY_COLORS.get(organ, (128, 128, 128))
        for c in range(3):
            out[:, :, c] = np.where(
                m,
                out[:, :, c] * (1 - alpha) + color[c] * alpha,
                out[:, :, c],
            )

    return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def save_slice_overlay(
    ct_image: sitk.Image,
    tissue_masks: dict[str, sitk.Image],
    output_path: str | Path,
    *,
    plane: str = "axial",
    slice_frac: float = 0.5,
    window_center: float = 40,
    window_width: float = 400,
    figsize: tuple[float, float] = (10, 10),
) -> Path:
    """Save a single CT slice with coloured organ overlays.

    Parameters
    ----------
    ct_image : CT volume (SimpleITK Image)
    tissue_masks : dict[organ_name, binary SimpleITK Image]
    output_path : PNG output file path
    plane : 'axial', 'coronal', or 'sagittal'
    slice_frac : fractional position (0-1) within the volume
    window_center, window_width : CT display windowing
    """
    plt = _safe_import_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ct_arr = sitk.GetArrayViewFromImage(ct_image)  # (z, y, x)
    zs, ys, xs = ct_arr.shape

    # Select slice
    if plane == "axial":
        idx = int(zs * slice_frac)
        ct_slice = ct_arr[idx, :, :]
        mask_slices = {
            k: sitk.GetArrayViewFromImage(v)[idx, :, :]
            for k, v in tissue_masks.items()
        }
        title = f"Axial Slice {idx}/{zs}"
    elif plane == "coronal":
        idx = int(ys * slice_frac)
        ct_slice = ct_arr[:, idx, :]
        mask_slices = {
            k: sitk.GetArrayViewFromImage(v)[:, idx, :]
            for k, v in tissue_masks.items()
        }
        title = f"Coronal Slice {idx}/{ys}"
    elif plane == "sagittal":
        idx = int(xs * slice_frac)
        ct_slice = ct_arr[:, :, idx]
        mask_slices = {
            k: sitk.GetArrayViewFromImage(v)[:, :, idx]
            for k, v in tissue_masks.items()
        }
        title = f"Sagittal Slice {idx}/{xs}"
    else:
        raise ValueError(f"Unknown plane: {plane}")

    # Apply CT windowing
    display = _ct_to_display(ct_slice.astype(np.float32),
                             window_center, window_width)
    ct_rgb = np.stack([display] * 3, axis=-1)

    # Blend overlays
    blended = _blend_overlay(ct_rgb, mask_slices)

    # Build colour legend
    organ_list = [k for k in tissue_masks if k in mask_slices]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(blended, origin="lower")
    ax.set_title(title, fontsize=14, color="white", pad=10)
    ax.axis("off")

    # Legend patches
    from matplotlib.patches import Patch
    patches = []
    for organ in organ_list:
        c = OVERLAY_COLORS.get(organ, (128, 128, 128))
        patches.append(Patch(
            facecolor=(c[0] / 255, c[1] / 255, c[2] / 255),
            label=organ.capitalize(),
        ))
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=10,
                  facecolor="black", edgecolor="gray", labelcolor="white")

    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)

    _log.info("Slice overlay saved: %s", output_path)
    return output_path


def save_montage(
    ct_image: sitk.Image,
    tissue_masks: dict[str, sitk.Image],
    output_path: str | Path,
    *,
    n_slices: int = 9,
    plane: str = "axial",
    window_center: float = 40,
    window_width: float = 400,
    figsize: tuple[float, float] = (18, 18),
) -> Path:
    """Save an NxN montage of evenly-spaced axial slices with overlays.

    Parameters
    ----------
    ct_image : CT volume
    tissue_masks : dict[organ_name, binary mask]
    output_path : PNG output path
    n_slices : number of slices in the montage
    """
    plt = _safe_import_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ct_arr = sitk.GetArrayViewFromImage(ct_image)
    zs, ys, xs = ct_arr.shape

    if plane == "axial":
        dim_size = zs
    elif plane == "coronal":
        dim_size = ys
    else:
        dim_size = xs

    cols = int(np.ceil(np.sqrt(n_slices)))
    rows = int(np.ceil(n_slices / cols))

    fracs = np.linspace(0.1, 0.9, n_slices)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for i, frac in enumerate(fracs):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]

        if plane == "axial":
            idx = int(zs * frac)
            ct_slice = ct_arr[idx, :, :]
            mask_slices = {
                k: sitk.GetArrayViewFromImage(v)[idx, :, :]
                for k, v in tissue_masks.items()
            }
            label = f"z={idx}"
        elif plane == "coronal":
            idx = int(ys * frac)
            ct_slice = ct_arr[:, idx, :]
            mask_slices = {
                k: sitk.GetArrayViewFromImage(v)[:, idx, :]
                for k, v in tissue_masks.items()
            }
            label = f"y={idx}"
        else:
            idx = int(xs * frac)
            ct_slice = ct_arr[:, :, idx]
            mask_slices = {
                k: sitk.GetArrayViewFromImage(v)[:, :, idx]
                for k, v in tissue_masks.items()
            }
            label = f"x={idx}"

        display = _ct_to_display(ct_slice.astype(np.float32),
                                 window_center, window_width)
        ct_rgb = np.stack([display] * 3, axis=-1)
        blended = _blend_overlay(ct_rgb, mask_slices)

        ax.imshow(blended, origin="lower")
        ax.set_title(label, fontsize=10, color="white")
        ax.axis("off")

    # Hide unused axes
    for j in range(len(fracs), len(axes_flat)):
        axes_flat[j].axis("off")

    # Add colour legend
    from matplotlib.patches import Patch
    patches = []
    for organ in tissue_masks:
        c = OVERLAY_COLORS.get(organ, (128, 128, 128))
        patches.append(Patch(
            facecolor=(c[0] / 255, c[1] / 255, c[2] / 255),
            label=organ.capitalize(),
        ))
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=len(patches),
                   fontsize=12, facecolor="black", edgecolor="gray",
                   labelcolor="white")

    fig.patch.set_facecolor("black")
    fig.suptitle(f"Segmentation Montage — {plane.capitalize()} ({n_slices} slices)",
                 fontsize=16, color="white", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)

    _log.info("Montage saved: %s (%d slices)", output_path, n_slices)
    return output_path


def save_segmentation_summary(
    ct_image: sitk.Image,
    tissue_masks: dict[str, sitk.Image],
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (20, 12),
) -> Path:
    """Save a combined visualization: axial + coronal + sagittal with overlays.

    Creates a 2×3 grid:
        Row 1: CT only (axial, coronal, sagittal at center)
        Row 2: CT with segmentation overlay
    """
    plt = _safe_import_plt()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ct_arr = sitk.GetArrayViewFromImage(ct_image)
    zs, ys, xs = ct_arr.shape

    planes = [
        ("Axial",    lambda a, i: a[i, :, :],               zs, 0.5),
        ("Coronal",  lambda a, i: a[:, i, :],               ys, 0.5),
        ("Sagittal", lambda a, i: a[:, :, i],               xs, 0.5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    for col, (label, slicer, dim, frac) in enumerate(planes):
        idx = int(dim * frac)
        ct_slice = slicer(ct_arr, idx).astype(np.float32)

        # CT only (Row 1) — bone window for better Overview
        display_bone = _ct_to_display(ct_slice, window_center=300, window_width=1500)
        ct_rgb_bone = np.stack([display_bone] * 3, axis=-1)
        axes[0][col].imshow(ct_rgb_bone, origin="lower", cmap="gray")
        axes[0][col].set_title(f"{label} — CT (z={idx})" if label == "Axial"
                               else f"{label} — CT", fontsize=11, color="white")
        axes[0][col].axis("off")

        # CT with overlay (Row 2) — soft tissue window
        display_soft = _ct_to_display(ct_slice, window_center=40, window_width=400)
        ct_rgb_soft = np.stack([display_soft] * 3, axis=-1)

        mask_slices = {}
        for organ, m in tissue_masks.items():
            mask_slices[organ] = slicer(sitk.GetArrayViewFromImage(m), idx)

        blended = _blend_overlay(ct_rgb_soft, mask_slices, alpha=0.4)
        axes[1][col].imshow(blended, origin="lower")
        axes[1][col].set_title(f"{label} — Segmentation", fontsize=11, color="white")
        axes[1][col].axis("off")

    # Legend
    from matplotlib.patches import Patch
    patches = [
        Patch(
            facecolor=(OVERLAY_COLORS.get(o, (128, 128, 128))[0] / 255,
                       OVERLAY_COLORS.get(o, (128, 128, 128))[1] / 255,
                       OVERLAY_COLORS.get(o, (128, 128, 128))[2] / 255),
            label=o.capitalize(),
        ) for o in tissue_masks
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=12, facecolor="black", edgecolor="gray",
               labelcolor="white")

    # Volume stats annotation
    spacing = ct_image.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]  # mm³
    stats_parts = []
    for organ, m in tissue_masks.items():
        count = int(np.count_nonzero(sitk.GetArrayViewFromImage(m)))
        vol_ml = count * voxel_vol / 1000.0  # mL
        stats_parts.append(f"{organ}: {vol_ml:.0f} mL")
    stats_text = "  |  ".join(stats_parts)

    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10,
             color="lightgray", fontfamily="monospace")

    fig.patch.set_facecolor("black")
    fig.suptitle("MedRecon — Segmentation Summary", fontsize=16,
                 color="white", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)

    _log.info("Segmentation summary saved: %s", output_path)
    return output_path
