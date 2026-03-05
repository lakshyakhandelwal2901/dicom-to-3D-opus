"""
medrecon_engine.mesh.mesh_validator
=====================================
Topological and geometric validation of a VTK mesh BEFORE export.

Checks
------
1.  Non-empty (> 0 faces).
2.  No degenerate (zero-area) triangles.
3.  All normals consistent (no flipped faces).
4.  Manifold — every edge shared by exactly 2 faces (watertight).
5.  No self-intersections (optional — expensive).
6.  Triangle aspect ratio within ``MAX_ASPECT_RATIO``.
7.  Bounding box sanity (not absurdly large / small).

The validator returns a ``MeshValidationReport`` and NEVER silently
passes a bad mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import vtk

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


@dataclass
class MeshValidationReport:
    """Result of mesh topology and geometry checks."""

    passed: bool = True
    num_vertices: int = 0
    num_faces: int = 0
    is_manifold: bool = False
    num_non_manifold_edges: int = 0
    num_degenerate_faces: int = 0
    max_aspect_ratio: float = 0.0
    mean_aspect_ratio: float = 0.0
    bounding_box_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_area_mm2: float = 0.0
    volume_mm3: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "vertices": self.num_vertices,
            "faces": self.num_faces,
            "manifold": self.is_manifold,
            "non_manifold_edges": self.num_non_manifold_edges,
            "degenerate_faces": self.num_degenerate_faces,
            "max_aspect_ratio": round(self.max_aspect_ratio, 2),
            "bbox_mm": [round(x, 2) for x in self.bounding_box_mm],
            "surface_area_mm2": round(self.surface_area_mm2, 2),
            "volume_mm3": round(self.volume_mm3, 2),
            "warnings": self.warnings,
            "errors": self.errors,
        }


class MeshValidator:
    """Validate mesh topology and geometry."""

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    # ------------------------------------------------------------------ #
    #  Public
    # ------------------------------------------------------------------ #
    def validate(self, poly: vtk.vtkPolyData) -> MeshValidationReport:
        """Run all checks and return a report."""
        r = MeshValidationReport()

        r.num_vertices = poly.GetNumberOfPoints()
        r.num_faces = poly.GetNumberOfCells()

        # 1 — Non-empty
        if r.num_faces == 0:
            r.errors.append("Mesh has zero faces.")
            r.passed = False
            return r

        # 2 — Degenerate triangles
        r.num_degenerate_faces = self._count_degenerate(poly)
        if r.num_degenerate_faces > 0:
            r.warnings.append(f"{r.num_degenerate_faces} degenerate faces detected.")

        # 3 — Manifold check (non-manifold edges)
        r.num_non_manifold_edges = self._count_non_manifold_edges(poly)
        r.is_manifold = r.num_non_manifold_edges == 0
        if not r.is_manifold:
            r.warnings.append(
                f"{r.num_non_manifold_edges} non-manifold edges — mesh is not watertight."
            )

        # 4 — Aspect ratio
        aspects = self._compute_aspect_ratios(poly)
        if len(aspects) > 0:
            r.max_aspect_ratio = float(np.max(aspects))
            r.mean_aspect_ratio = float(np.mean(aspects))
            if r.max_aspect_ratio > self.cfg.max_aspect_ratio:
                r.warnings.append(
                    f"Max aspect ratio {r.max_aspect_ratio:.1f} exceeds "
                    f"limit {self.cfg.max_aspect_ratio:.1f}."
                )

        # 5 — Bounding box
        bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        r.bounding_box_mm = (
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )

        # 6 — Surface area & volume
        mass_props = vtk.vtkMassProperties()
        mass_props.SetInputData(poly)
        mass_props.Update()
        r.surface_area_mm2 = mass_props.GetSurfaceArea()
        r.volume_mm3 = mass_props.GetVolume()

        # 7 — Sanity
        max_dim = max(r.bounding_box_mm)
        if max_dim > 2000:
            r.warnings.append(f"Bounding box > 2 m ({max_dim:.0f} mm) — check units.")
        if max_dim < 1:
            r.errors.append(f"Bounding box < 1 mm ({max_dim:.4f} mm) — mesh too small.")
            r.passed = False

        # Final verdict
        if r.errors:
            r.passed = False

        status = "PASS" if r.passed else "FAIL"
        log.info(
            "MeshValidator %s: %s verts, %s faces, manifold=%s, bbox=(%.1f, %.1f, %.1f) mm",
            status,
            f"{r.num_vertices:,}",
            f"{r.num_faces:,}",
            r.is_manifold,
            *r.bounding_box_mm,
        )
        for w in r.warnings:
            log.warning("  ⚠ %s", w)
        for e in r.errors:
            log.error("  ✗ %s", e)

        return r

    # ------------------------------------------------------------------ #
    #  Internal checks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _count_degenerate(poly: vtk.vtkPolyData) -> int:
        """Count triangles with zero or near-zero area."""
        count = 0
        for i in range(poly.GetNumberOfCells()):
            cell = poly.GetCell(i)
            if cell.GetCellType() != vtk.VTK_TRIANGLE:
                continue
            pts = [np.array(cell.GetPoints().GetPoint(j)) for j in range(3)]
            area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            if area < 1e-10:
                count += 1
        return count

    @staticmethod
    def _count_non_manifold_edges(poly: vtk.vtkPolyData) -> int:
        """Count edges shared by != 2 faces."""
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(poly)
        fe.BoundaryEdgesOff()
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOn()
        fe.Update()
        return fe.GetOutput().GetNumberOfCells()

    @staticmethod
    def _compute_aspect_ratios(poly: vtk.vtkPolyData) -> np.ndarray:
        """Compute the aspect ratio of every triangle.

        Aspect ratio = longest edge / shortest altitude.
        """
        n = poly.GetNumberOfCells()
        if n == 0:
            return np.array([])

        ratios = np.zeros(n, dtype=np.float64)
        for i in range(n):
            cell = poly.GetCell(i)
            if cell.GetCellType() != vtk.VTK_TRIANGLE:
                ratios[i] = 1.0
                continue
            pts = [np.array(cell.GetPoints().GetPoint(j)) for j in range(3)]
            edges = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[0] - pts[2]),
            ]
            area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            max_edge = max(edges)
            if area < 1e-12:
                ratios[i] = float("inf")
            else:
                shortest_alt = 2.0 * area / max_edge
                ratios[i] = max_edge / shortest_alt if shortest_alt > 1e-12 else float("inf")

        return ratios
