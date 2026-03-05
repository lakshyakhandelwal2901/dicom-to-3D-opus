"""
medrecon_engine.export.obj_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Write VTK ``vtkPolyData`` meshes to **OBJ** files.

Provides both single-mesh and grouped multi-mesh export:

* ``save_obj``         — one mesh → one OBJ file
* ``save_grouped_obj`` — dict of meshes → one OBJ file with named groups

Uses ``vtkOBJWriter`` for single exports and a manual writer for grouped
output (OBJ-aware viewers can toggle groups on/off).
"""

from __future__ import annotations

from pathlib import Path

import vtk

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)


def save_obj(mesh: vtk.vtkPolyData, filepath: str | Path) -> Path:
    """Export a single VTK mesh as an OBJ file.

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Triangulated surface mesh.
    filepath : str | Path
        Destination ``.obj`` path.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(filepath))
    writer.SetInputData(mesh)
    writer.Write()

    _log.info(
        "OBJ → %s  (%d pts, %d faces)",
        filepath.name,
        mesh.GetNumberOfPoints(),
        mesh.GetNumberOfCells(),
    )
    return filepath.resolve()


def save_grouped_obj(
    meshes: dict[str, vtk.vtkPolyData],
    filepath: str | Path,
) -> Path:
    """Write a single OBJ with named groups.

    Parameters
    ----------
    meshes : dict[str, vtkPolyData]
        Mapping from group name → mesh.
    filepath : str | Path
        Destination ``.obj`` path.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    vertex_offset = 0

    with open(filepath, "w") as fh:
        fh.write(f"# MedRecon Engine — grouped OBJ\n")
        fh.write(f"# Groups: {', '.join(meshes.keys())}\n\n")

        for group_name, poly in meshes.items():
            fh.write(f"g {group_name}\n")

            pts = poly.GetPoints()
            n_pts = pts.GetNumberOfPoints()
            n_cells = poly.GetNumberOfCells()

            for i in range(n_pts):
                x, y, z = pts.GetPoint(i)
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

            pn = poly.GetPointData().GetNormals()
            has_normals = pn is not None
            if has_normals:
                for i in range(n_pts):
                    nx, ny, nz = pn.GetTuple3(i)
                    fh.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

            for i in range(n_cells):
                cell = poly.GetCell(i)
                n_cell_pts = cell.GetNumberOfPoints()
                if n_cell_pts < 3:
                    continue
                ids = [cell.GetPointId(j) + 1 + vertex_offset for j in range(n_cell_pts)]
                if has_normals:
                    fh.write("f " + " ".join(f"{v}//{v}" for v in ids) + "\n")
                else:
                    fh.write("f " + " ".join(str(v) for v in ids) + "\n")

            vertex_offset += n_pts

    _log.info("Grouped OBJ → %s  (%d groups)", filepath.name, len(meshes))
    return filepath.resolve()
