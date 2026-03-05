"""
medrecon_engine.mesh.mesh_postprocess
=======================================
Clean and optimise a raw VTK mesh for surgical / 3D-printing use.

Pipeline (executed in order):
1.  **Clean** — remove duplicate points & degenerate cells.
2.  **Smooth** — Windowed Sinc (Taubin) smoothing to remove marching-cubes
    staircase artefacts WITHOUT shrinkage (unlike Laplacian).
3.  **Decimate** — Quadric clustering decimation to reduce face count while
    preserving sharp bone edges.
4.  **Normals** — Recompute consistent outward-facing normals.
5.  **Fill holes** — Plug small boundary loops so the mesh is watertight.

Every step is individually toggleable via ``PrecisionConfig`` values.
"""

from __future__ import annotations

import vtk

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class MeshPostProcessor:
    """Post-process raw VTK mesh to surgical / print quality."""

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    # ------------------------------------------------------------------ #
    #  Public
    # ------------------------------------------------------------------ #
    def process(self, poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Apply full post-processing pipeline to *poly*."""
        initial_faces = poly.GetNumberOfCells()
        log.info("PostProcess: input — %s faces", f"{initial_faces:,}")

        # 1 — Clean
        poly = self._clean(poly)

        # 2 — Smooth (Taubin / windowed sinc)
        if self.cfg.mesh_smooth_iterations > 0:
            poly = self._smooth(poly)

        # 3 — Decimate
        if 0 < self.cfg.mesh_decimate_target_ratio < 1.0:
            poly = self._decimate(poly)

        # 4 — Recompute normals
        poly = self._normals(poly)

        # 5 — Fill holes
        poly = self._fill_holes(poly)

        final_faces = poly.GetNumberOfCells()
        log.info(
            "PostProcess: done — %s → %s faces",
            f"{initial_faces:,}", f"{final_faces:,}",
        )
        return poly

    # ------------------------------------------------------------------ #
    #  Steps
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(poly)
        cleaner.ConvertLinesToPointsOn()
        cleaner.ConvertPolysToLinesOn()
        cleaner.ConvertStripsToPolysOn()
        cleaner.PointMergingOn()
        cleaner.Update()
        return cleaner.GetOutput()

    def _smooth(self, poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Windowed Sinc smoothing (Taubin — no shrinkage)."""
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(poly)
        smoother.SetNumberOfIterations(self.cfg.mesh_smooth_iterations)
        smoother.SetPassBand(self.cfg.mesh_smooth_passband)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        log.info("  Smoothed: %d iterations, passband=%.4f",
                 self.cfg.mesh_smooth_iterations, self.cfg.mesh_smooth_passband)
        return smoother.GetOutput()

    def _decimate(self, poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Quadric decimation preserving topology."""
        before = poly.GetNumberOfCells()
        reduction = 1.0 - self.cfg.mesh_decimate_target_ratio

        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(poly)
        decimator.SetTargetReduction(reduction)
        decimator.VolumePreservationOn()
        decimator.Update()

        after = decimator.GetOutput().GetNumberOfCells()
        log.info("  Decimated: %s → %s faces (%.0f%% reduction)",
                 f"{before:,}", f"{after:,}", reduction * 100)
        return decimator.GetOutput()

    @staticmethod
    def _normals(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Recompute consistent normals."""
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(poly)
        norms.ConsistencyOn()
        norms.AutoOrientNormalsOn()
        norms.SplittingOff()
        norms.Update()
        return norms.GetOutput()

    def _fill_holes(self, poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Fill small boundary loops."""
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(poly)
        filler.SetHoleSize(self.cfg.mesh_fill_holes_size)
        filler.Update()

        filled = filler.GetOutput()

        # Re-norm after filling
        filled = self._normals(filled)
        return filled
