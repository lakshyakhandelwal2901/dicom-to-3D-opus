"""
medrecon_engine.quality.surface_metrics
=========================================
Compute quantitative surface-quality metrics on a VTK mesh.

Metrics
-------
* **Face count** — proxy for resolution / level of detail.
* **Surface area** (mm²).
* **Volume** (mm³) — meaningful only if mesh is manifold.
* **Mean / max / std edge length** — reveals over- / under-decimation.
* **Mean / max / std aspect ratio** — reveals skinny triangles.
* **Hausdorff-approximated smoothness** — not a true Hausdorff, but the
  standard deviation of face normals in a local neighbourhood, which
  indicates staircase artefact remnants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import vtk

from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


@dataclass
class SurfaceMetrics:
    """Quantitative mesh quality report."""

    num_vertices: int = 0
    num_faces: int = 0

    surface_area_mm2: float = 0.0
    volume_mm3: float = 0.0

    edge_length_mean: float = 0.0
    edge_length_max: float = 0.0
    edge_length_std: float = 0.0

    aspect_ratio_mean: float = 0.0
    aspect_ratio_max: float = 0.0
    aspect_ratio_std: float = 0.0

    normal_consistency: float = 1.0    # 1.0 = perfect, 0 = chaotic

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


class SurfaceMetricsComputer:
    """Compute surface quality metrics on a vtkPolyData mesh."""

    def compute(self, poly: vtk.vtkPolyData) -> SurfaceMetrics:
        m = SurfaceMetrics()
        m.num_vertices = poly.GetNumberOfPoints()
        m.num_faces = poly.GetNumberOfCells()

        if m.num_faces == 0:
            return m

        # Surface area + volume
        mass = vtk.vtkMassProperties()
        mass.SetInputData(poly)
        mass.Update()
        m.surface_area_mm2 = mass.GetSurfaceArea()
        m.volume_mm3 = mass.GetVolume()

        # Edge lengths + aspect ratios
        edges_all: list[float] = []
        aspects_all: list[float] = []

        for i in range(m.num_faces):
            cell = poly.GetCell(i)
            if cell.GetCellType() != vtk.VTK_TRIANGLE:
                continue
            pts = [np.array(cell.GetPoints().GetPoint(j)) for j in range(3)]
            e = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[0] - pts[2]),
            ]
            edges_all.extend(e)

            area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            max_e = max(e)
            if area > 1e-12:
                alt = 2.0 * area / max_e if max_e > 1e-12 else 1e-12
                aspects_all.append(max_e / alt)
            else:
                aspects_all.append(float("inf"))

        if edges_all:
            ea = np.array(edges_all)
            m.edge_length_mean = float(np.mean(ea))
            m.edge_length_max = float(np.max(ea))
            m.edge_length_std = float(np.std(ea))

        if aspects_all:
            finite = np.array([a for a in aspects_all if np.isfinite(a)])
            if len(finite) > 0:
                m.aspect_ratio_mean = float(np.mean(finite))
                m.aspect_ratio_max = float(np.max(finite))
                m.aspect_ratio_std = float(np.std(finite))

        # Normal consistency
        m.normal_consistency = self._normal_consistency(poly)

        log.info(
            "SurfaceMetrics: area=%.0f mm²  vol=%.0f mm³  "
            "edge=%.2f±%.2f mm  aspect=%.2f (max %.1f)",
            m.surface_area_mm2, m.volume_mm3,
            m.edge_length_mean, m.edge_length_std,
            m.aspect_ratio_mean, m.aspect_ratio_max,
        )
        return m

    # ------------------------------------------------------------------ #
    #  Normal consistency
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normal_consistency(poly: vtk.vtkPolyData) -> float:
        """Estimate how consistent face normals are within local patches.

        Returns a value in [0, 1] where 1.0 = perfectly smooth normals.
        """
        normals = poly.GetCellData().GetNormals()
        if normals is None:
            # recompute
            nf = vtk.vtkPolyDataNormals()
            nf.SetInputData(poly)
            nf.ComputeCellNormalsOn()
            nf.Update()
            poly = nf.GetOutput()
            normals = poly.GetCellData().GetNormals()
            if normals is None:
                return 0.0

        n = normals.GetNumberOfTuples()
        if n < 2:
            return 1.0

        # Compute mean direction and measure spread
        normal_arr = np.array([normals.GetTuple3(i) for i in range(n)])
        norms = np.linalg.norm(normal_arr, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normal_arr = normal_arr / norms

        # Variance of normals as consistency metric
        var = float(np.mean(np.var(normal_arr, axis=0)))
        # var ∈ [0, ~0.67] — map to [1, 0]
        consistency = max(0.0, 1.0 - var * 1.5)
        return consistency
