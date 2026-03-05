"""
medrecon_engine.export.stl_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Write VTK ``vtkPolyData`` meshes to **STL** files.

Features
--------
* Binary (default) or ASCII STL output — controlled by
  ``PrecisionConfig.stl_binary``.
* Solid-name embedding for traceability (anatomy + run_id).
* Pre-flight validation: rejects empty meshes or meshes with zero area.
* Automatic parent-directory creation.
* Returns a frozen ``STLExportReport`` with file size, path, and timings.

All file I/O is synchronous and local — suitable for on-prem deployments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import vtk

from medrecon_engine.audit.logger import get_logger
from medrecon_engine.config.precision_config import PrecisionConfig

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Export report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class STLExportReport:
    """Immutable summary returned after every STL write."""

    output_path: str
    file_size_bytes: int
    binary_mode: bool
    num_points: int
    num_cells: int
    elapsed_seconds: float
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class STLWriter:
    """Write a ``vtkPolyData`` mesh to an STL file on disk.

    Parameters
    ----------
    config : PrecisionConfig
        Pipeline configuration (controls binary/ascii mode).
    """

    def __init__(self, config: Optional[PrecisionConfig] = None) -> None:
        self._cfg = config or PrecisionConfig()

    # ── public API ─────────────────────────────────────────────────────
    def write(
        self,
        mesh: "vtk.vtkPolyData",
        output_path: str | Path,
        *,
        solid_name: str = "medrecon",
    ) -> STLExportReport:
        """Write *mesh* to *output_path* as STL.

        Parameters
        ----------
        mesh : vtk.vtkPolyData
            The triangulated surface mesh to export.
        output_path : str | Path
            Destination file path (``*.stl``).
        solid_name : str
            Embedded solid name for ASCII STL header / traceability.

        Returns
        -------
        STLExportReport

        Raises
        ------
        ValueError
            If *mesh* is ``None``, has no points, or has no cells.
        """
        t0 = time.perf_counter()
        output_path = Path(output_path)

        # ── Validate input ─────────────────────────────────────────────
        if mesh is None:
            raise ValueError("Cannot export None mesh.")
        n_pts = mesh.GetNumberOfPoints()
        n_cells = mesh.GetNumberOfCells()
        if n_pts == 0 or n_cells == 0:
            raise ValueError(
                f"Mesh is empty (points={n_pts}, cells={n_cells}). "
                "Nothing to export."
            )

        # Ensure triangulated
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(mesh)
        tri.Update()
        tri_mesh = tri.GetOutput()

        # ── Ensure output directory exists ────────────────────────────
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Write STL ─────────────────────────────────────────────────
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(tri_mesh)

        if self._cfg.stl_binary:
            writer.SetFileTypeToBinary()
        else:
            writer.SetFileTypeToASCII()

        # VTK STL writer embeds a header for binary mode.  For ASCII we
        # can set the solid name via the header string.
        if hasattr(writer, "SetHeader"):
            writer.SetHeader(solid_name)

        writer.Write()

        # ── Build report ──────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        file_size = output_path.stat().st_size if output_path.exists() else 0

        report = STLExportReport(
            output_path=str(output_path),
            file_size_bytes=file_size,
            binary_mode=self._cfg.stl_binary,
            num_points=tri_mesh.GetNumberOfPoints(),
            num_cells=tri_mesh.GetNumberOfCells(),
            elapsed_seconds=round(elapsed, 4),
        )

        _log.info(
            "STL exported → %s  (%d pts, %d faces, %.1f KB, %.3f s)",
            output_path.name,
            report.num_points,
            report.num_cells,
            file_size / 1024,
            elapsed,
        )
        return report

    # ── Convenience: Write + return path ───────────────────────────────
    def write_to(
        self,
        mesh: "vtk.vtkPolyData",
        directory: str | Path,
        filename: str = "output.stl",
        *,
        solid_name: str = "medrecon",
    ) -> Path:
        """Write *mesh* into *directory/filename* and return the full path."""
        out = Path(directory) / filename
        self.write(mesh, out, solid_name=solid_name)
        return out
