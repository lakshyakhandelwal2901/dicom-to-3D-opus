"""
medrecon_engine.mesh.mesh_organizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Organize and merge individual AI-segmented meshes into a clean,
categorised folder structure.

Responsibilities
----------------
1. **Categorise** — route each label into bones / organs / lungs /
   vessels / muscles / others.
2. **Merge** — combine related structures (e.g. all vertebrae → spine,
   left + right kidney → kidneys).
3. **Export** — write final OBJ files into the organised tree.

Typical output::

    output/
        bones/
            spine.obj
            ribs.obj
            pelvis.obj
            femur_left.obj
            femur_right.obj
        organs/
            liver.obj
            kidneys.obj
            spleen.obj
            …
        lungs/
            lungs.obj
        vessels/
            aorta.obj
            …
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import vtk

from medrecon_engine.audit.logger import get_logger
from medrecon_engine.config.structure_groups import (
    get_category,
    get_merge_target,
)
from medrecon_engine.mesh.vtk_generator import merge_meshes
from medrecon_engine.export.obj_writer import save_obj

_log = get_logger(__name__)


def organize_and_merge(
    meshes: dict[str, vtk.vtkPolyData],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Categorise, merge, and export meshes into an organised tree.

    Parameters
    ----------
    meshes : dict[str, vtkPolyData]
        Mapping  label_name → mesh  (e.g. from the label-to-mesh step).
    output_dir : str | Path
        Root output directory.  Sub-folders are created automatically.

    Returns
    -------
    dict[str, Path]
        Mapping  final_name → written OBJ path.
    """
    output_dir = Path(output_dir)
    t0 = time.perf_counter()

    # ── 1. Bucket meshes by merge target ──────────────────────────────
    merge_buckets: dict[str, list[tuple[str, vtk.vtkPolyData]]] = defaultdict(list)
    standalone: dict[str, vtk.vtkPolyData] = {}

    for label_name, mesh in meshes.items():
        target = get_merge_target(label_name)
        if target:
            merge_buckets[target].append((label_name, mesh))
        else:
            standalone[label_name] = mesh

    _log.info(
        "Organizer: %d labels → %d merge groups + %d standalone",
        len(meshes),
        len(merge_buckets),
        len(standalone),
    )

    # ── 2. Merge groups ───────────────────────────────────────────────
    merged: dict[str, vtk.vtkPolyData] = {}
    for target_name, items in merge_buckets.items():
        parts = [m for _, m in items]
        _log.info(
            "  Merging %d labels → %s  (%s)",
            len(parts),
            target_name,
            ", ".join(n for n, _ in items),
        )
        merged[target_name] = merge_meshes(parts)

    # ── 3. Combine merged + standalone ────────────────────────────────
    all_meshes: dict[str, vtk.vtkPolyData] = {}
    all_meshes.update(merged)
    all_meshes.update(standalone)

    # ── 4. Write into categorised folders ─────────────────────────────
    results: dict[str, Path] = {}

    for name, mesh in sorted(all_meshes.items()):
        category = get_category(name)
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        obj_path = save_obj(mesh, cat_dir / f"{name}.obj")
        results[name] = obj_path

    elapsed = time.perf_counter() - t0
    _log.info(
        "Organizer complete: %d OBJ files in %.1f s",
        len(results),
        elapsed,
    )

    # Summary per category
    from collections import Counter
    cats = Counter(get_category(n) for n in results)
    for cat, count in sorted(cats.items()):
        _log.info("  %s/ → %d meshes", cat, count)

    return results
