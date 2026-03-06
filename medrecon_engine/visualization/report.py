"""
medrecon_engine.visualization.report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a full HTML report with embedded images for a
segmentation run.

The report includes:
- 3D multi-view screenshot
- Segmentation summary (axial/coronal/sagittal)
- Axial montage
- Volume statistics table
- Pipeline parameters

Usage::

    from medrecon_engine.visualization.report import generate_report
    generate_report(ct_image, tissue_masks, vtk_meshes, output_dir)
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)


def _img_to_base64(path: Path) -> str:
    """Read a PNG file and return a base64-encoded data URI."""
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def generate_report(
    ct_image: sitk.Image,
    tissue_masks: dict[str, sitk.Image],
    vtk_meshes: dict,
    output_dir: str | Path,
    *,
    patient_id: str = "Unknown",
    pipeline_time: float = 0.0,
) -> Path:
    """Generate a full HTML report with embedded visualizations.

    Parameters
    ----------
    ct_image : Original CT volume
    tissue_masks : dict[organ, binary mask]
    vtk_meshes : dict[organ, vtkPolyData]
    output_dir : Directory for output files
    patient_id : Patient identifier for the report header
    pipeline_time : Total pipeline execution time (seconds)

    Returns
    -------
    Path to the generated HTML report
    """
    from medrecon_engine.visualization.render3d import save_multi_view
    from medrecon_engine.visualization.slice_overlay import (
        save_montage,
        save_segmentation_summary,
    )

    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Generating visualization report …")
    t0 = time.perf_counter()

    # ── Generate images ────────────────────────────────────────────
    multi_view_path = save_multi_view(vtk_meshes, viz_dir / "3d_views.png")
    summary_path = save_segmentation_summary(
        ct_image, tissue_masks, viz_dir / "seg_summary.png"
    )
    montage_path = save_montage(
        ct_image, tissue_masks, viz_dir / "montage.png", n_slices=9
    )

    # ── Volume statistics ──────────────────────────────────────────
    spacing = ct_image.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    stats_rows = []
    for organ, mask in tissue_masks.items():
        count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
        vol_ml = count * voxel_vol_mm3 / 1000.0
        mesh = vtk_meshes.get(organ)
        n_pts = mesh.GetNumberOfPoints() if mesh else 0
        n_faces = mesh.GetNumberOfCells() if mesh else 0
        stats_rows.append({
            "organ": organ.capitalize(),
            "voxels": f"{count:,}",
            "volume_ml": f"{vol_ml:.1f}",
            "vertices": f"{n_pts:,}",
            "faces": f"{n_faces:,}",
        })

    # ── Build HTML ─────────────────────────────────────────────────
    mv_b64 = _img_to_base64(multi_view_path)
    sum_b64 = _img_to_base64(summary_path)
    mont_b64 = _img_to_base64(montage_path)

    table_rows_html = "\n".join(
        f"""<tr>
            <td>{r['organ']}</td>
            <td>{r['voxels']}</td>
            <td>{r['volume_ml']} mL</td>
            <td>{r['vertices']}</td>
            <td>{r['faces']}</td>
        </tr>"""
        for r in stats_rows
    )

    size = ct_image.GetSize()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MedRecon — Segmentation Report</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, sans-serif;
        background: #0d1117;
        color: #e6edf3;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }}
    h1 {{
        color: #58a6ff;
        border-bottom: 2px solid #30363d;
        padding-bottom: 10px;
    }}
    h2 {{
        color: #79c0ff;
        margin-top: 30px;
    }}
    .meta {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 15px 0;
        font-family: monospace;
        font-size: 14px;
    }}
    .meta span {{
        color: #8b949e;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #30363d;
    }}
    th {{
        background: #161b22;
        color: #58a6ff;
        font-weight: 600;
    }}
    tr:hover {{
        background: #161b22;
    }}
    img {{
        max-width: 100%;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #30363d;
    }}
    .section {{
        margin: 25px 0;
    }}
    .footer {{
        text-align: center;
        color: #484f58;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 15px;
        border-top: 1px solid #21262d;
    }}
</style>
</head>
<body>

<h1>MedRecon — Segmentation Report</h1>

<div class="meta">
    <span>Patient:</span> {patient_id}<br>
    <span>Volume:</span> {size[0]} x {size[1]} x {size[2]}
    &nbsp;|&nbsp;
    <span>Spacing:</span> {spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm<br>
    <span>Organs:</span> {', '.join(m.capitalize() for m in tissue_masks)}<br>
    <span>Pipeline time:</span> {pipeline_time:.1f} s
</div>

<div class="section">
    <h2>3D Multi-View</h2>
    <img src="{mv_b64}" alt="3D multi-view rendering">
</div>

<div class="section">
    <h2>Segmentation Summary</h2>
    <img src="{sum_b64}" alt="Segmentation summary">
</div>

<div class="section">
    <h2>Axial Montage</h2>
    <img src="{mont_b64}" alt="Axial montage">
</div>

<div class="section">
    <h2>Volume Statistics</h2>
    <table>
        <tr>
            <th>Organ</th>
            <th>Voxels</th>
            <th>Volume</th>
            <th>Vertices</th>
            <th>Faces</th>
        </tr>
        {table_rows_html}
    </table>
</div>

<div class="footer">
    Generated by MedRecon Engine &mdash; CT to Surgical 3D Pipeline
</div>

</body>
</html>"""

    report_path = output_dir / "segmentation_report.html"
    report_path.write_text(html, encoding="utf-8")

    dt = time.perf_counter() - t0
    _log.info("Report generated in %.1f s: %s", dt, report_path)
    return report_path
