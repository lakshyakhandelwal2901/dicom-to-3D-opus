"""
MedRecon Engine — Triple OBJ Pipeline  (v4 — clean separation)
=================================================================
Produces a single grouped OBJ from a CT DICOM volume:

    human_model.obj
        g bones          → clean skeleton (HU>200, top-15 CC, morph close)
        g organs          → gradient boundaries, bones subtracted, skin eroded
        g body_surface    → outer skin envelope (largest CC of HU > -300)

Bone cleaning:
    1. HU > 200              (removes soft tissue / calcified cartilage)
    2. ConnectedComponent     keep largest 15 structures (spine, ribs, pelvis …)
    3. BinaryMorphologicalClosing  fill internal gaps

Organ segmentation — gradient + subtraction:
    1. Gradient pipeline     (window → Gaussian → GradientMagnitude → edge mask)
    2. Subtract bones        remove HU > 200
    3. Erode skin shell      BinaryErode body exterior
    4. Keep interior only    intersect with eroded body
    5. CC filter             keep largest 8 structures
    6. Region grow           ConnectedThreshold from body centre

Mesh extraction uses vtkFlyingEdges3D (faster, smoother than marching cubes).
Post-processing: FillHoles → Smooth (Taubin) → Decimate → Normals

Also writes individual files: bones.obj, organs.obj, body_surface.obj

Usage
-----
::

    python triple_obj_pipeline.py --dicom D:\\DATASET\\...\\L067 --output ./output
    python triple_obj_pipeline.py --dicom D:\\DATASET\\...\\Pelvic-Ref-001 --output ./output --max-slice 5.0
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import SimpleITK as sitk
import vtk
from rich.console import Console
from rich.table import Table

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.core.dataset_scanner import DatasetScanner
from medrecon_engine.core.dicom_validator import DicomValidator
from medrecon_engine.core.volume_loader import VolumeLoader
from medrecon_engine.core.hu_converter import HUConverter
from medrecon_engine.mesh.vtk_generator import generate_mesh

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

VOXEL_DOWNSAMPLE_THRESHOLD = 15_000_000
FACE_PREDECIMATE_THRESHOLD = 5_000_000


# ═══════════════════════════════════════════════════════════════════════════
# Mask / segmentation helpers
# ═══════════════════════════════════════════════════════════════════════════

def mask_from_threshold(hu_volume: sitk.Image, lower: float,
                        upper: float = 3001.0) -> sitk.Image:
    """Binary mask via HU window, preserving spatial metadata."""
    arr = sitk.GetArrayFromImage(hu_volume).astype(np.float32)
    mask_arr = ((arr > lower) & (arr < upper)).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(hu_volume)
    return mask


def segment_bones_clean(
    hu_volume: sitk.Image,
    *,
    hu_threshold: float = 200.0,
    max_components: int = 15,
) -> sitk.Image:
    """Clean bone segmentation — skeleton only, no soft tissue.

    Steps:
        1. HU > 200  (higher than 150 → removes calcified cartilage,
           contrast material, dense organs)
        2. ConnectedComponent → RelabelComponent → keep largest *max_components*
           (spine, ribs, pelvis, femur — discard small fragments)
        3. BinaryMorphologicalClosing [2,2,2] → fill internal gaps
    """
    # Step 1 — high HU threshold
    mask = mask_from_threshold(hu_volume, lower=hu_threshold)

    # Step 2 — keep only the largest connected structures
    cc = sitk.ConnectedComponent(mask)
    relabelled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    # Labels 1…max_components are the largest; discard the rest
    mask_clean = sitk.BinaryThreshold(
        relabelled, lowerThreshold=1, upperThreshold=max_components,
    )
    mask_clean.CopyInformation(hu_volume)

    # Step 3 — morphological closing to fill gaps
    mask_clean = sitk.BinaryMorphologicalClosing(mask_clean, [2, 2, 2])
    mask_clean.CopyInformation(hu_volume)
    return sitk.Cast(mask_clean, sitk.sitkUInt8)


def segment_organs_gradient(
    hu_volume: sitk.Image,
    *,
    gauss_variance: float = 1.5,
    gradient_threshold: float = 25.0,
    hu_lower: float = -200.0,
    hu_upper: float = 300.0,
    bone_threshold: float = 200.0,
    skin_erode_radius: int = 5,
    max_organ_components: int = 8,
) -> sitk.Image:
    """Gradient + subtraction organ segmentation — no bones, no skin.

    Pipeline:
        1. HU window    → clip to [-300, 500]
        2. Gaussian     → DiscreteGaussian(variance) removes CT speckle
        3. Gradient mag → highlights organ/tissue boundaries
        4. Edge mask    → gradient > threshold
        5. HU candidate → hu_lower < HU < hu_upper (excludes air & bone cores)
        6. Combine      → candidate AND edge_mask
        7. Subtract bones  → remove HU > bone_threshold
        8. Erode skin      → BinaryErode body exterior, keep interior only
        9. CC filter       → keep largest max_organ_components structures
       10. Region grow     → ConnectedThreshold from body centre
    """
    vol_f32 = sitk.Cast(hu_volume, sitk.sitkFloat32)

    # Step 1 — HU window
    arr = sitk.GetArrayFromImage(vol_f32).copy()
    arr = np.clip(arr, -300, 500).astype(np.float32)
    windowed = sitk.GetImageFromArray(arr)
    windowed.CopyInformation(hu_volume)
    windowed = sitk.Cast(windowed, sitk.sitkFloat32)

    # Step 2 — Gaussian smooth
    smoothed = sitk.DiscreteGaussian(windowed, variance=gauss_variance)

    # Step 3 — Gradient magnitude
    gradient = sitk.GradientMagnitude(smoothed)

    # Step 4 — Edge mask
    grad_arr = sitk.GetArrayFromImage(gradient)
    edge_arr = (grad_arr > gradient_threshold).astype(np.uint8)
    edge_mask = sitk.GetImageFromArray(edge_arr)
    edge_mask.CopyInformation(hu_volume)

    # Step 5 — HU candidate mask
    candidate = mask_from_threshold(hu_volume, lower=hu_lower, upper=hu_upper)

    # Step 6 — Combine: boundary voxels in tissue window
    combined = sitk.And(candidate, edge_mask)

    # Step 7 — Subtract bones (HU > bone_threshold)
    bone_mask = mask_from_threshold(hu_volume, lower=bone_threshold)
    combined = sitk.And(combined, sitk.InvertIntensity(bone_mask, maximum=1))

    # Step 8 — Erode skin shell: detect body → erode outer layer → keep interior
    body = mask_from_threshold(hu_volume, lower=-300)
    body = sitk.BinaryMorphologicalClosing(body, [3, 3, 3])
    body.CopyInformation(hu_volume)
    radius = [skin_erode_radius] * 3
    inner_body = sitk.BinaryErode(body, radius)
    inner_body.CopyInformation(hu_volume)
    combined = sitk.And(combined, sitk.Cast(inner_body, sitk.sitkUInt8))

    # Step 9 — CC filter: keep largest organ structures
    cc = sitk.ConnectedComponent(combined)
    relabelled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    combined = sitk.BinaryThreshold(
        relabelled, lowerThreshold=1, upperThreshold=max_organ_components,
    )
    combined.CopyInformation(hu_volume)

    # Step 10 — Region grow from body centre to keep main connected mass
    sz = combined.GetSize()  # (x, y, z)
    cx, cy, cz = sz[0] // 2, sz[1] // 2, sz[2] // 2

    seeds = []
    offsets = [-10, -5, 0, 5, 10]
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                sx = max(0, min(sz[0] - 1, cx + dx))
                sy = max(0, min(sz[1] - 1, cy + dy))
                sz_ = max(0, min(sz[2] - 1, cz + dz))
                seeds.append((sx, sy, sz_))

    region = sitk.ConnectedThreshold(
        sitk.Cast(combined, sitk.sitkUInt8),
        seedList=seeds,
        lower=1,
        upper=1,
    )
    region.CopyInformation(hu_volume)
    return sitk.Cast(region, sitk.sitkUInt8)


def segment_body_surface(hu_volume: sitk.Image) -> sitk.Image:
    """Extract outer body surface — largest connected component of HU > -300.

    Gives the skin envelope rather than all internal structures.
    """
    mask = mask_from_threshold(hu_volume, lower=-300)

    # Largest connected component — keeps the body, discards table/noise
    labelled = sitk.ConnectedComponent(mask)
    labelled = sitk.RelabelComponent(labelled, sortByObjectSize=True)
    body = sitk.BinaryThreshold(labelled, lowerThreshold=1, upperThreshold=1)
    body.CopyInformation(hu_volume)

    # Morphological closing to smooth the skin
    body = sitk.BinaryMorphologicalClosing(body, [3, 3, 3])
    body.CopyInformation(hu_volume)
    return sitk.Cast(body, sitk.sitkUInt8)


# ═══════════════════════════════════════════════════════════════════════════
# Large-mesh helpers
# ═══════════════════════════════════════════════════════════════════════════

def downsample_mask(mask: sitk.Image, factor: int = 2) -> sitk.Image:
    """Downsample binary mask by *factor* (nearest-neighbor)."""
    original_size = mask.GetSize()
    original_spacing = mask.GetSpacing()
    new_spacing = [s * factor for s in original_spacing]
    new_size = [max(1, int(round(s / factor))) for s in original_size]
    return sitk.Resample(
        mask, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
        mask.GetOrigin(), new_spacing, mask.GetDirection(), 0, mask.GetPixelID(),
    )


def fast_predecimate(poly: vtk.vtkPolyData, reduction: float = 0.80) -> vtk.vtkPolyData:
    """Fast coarse pass using vtkDecimatePro before expensive smooth."""
    dec = vtk.vtkDecimatePro()
    dec.SetInputData(poly)
    dec.SetTargetReduction(reduction)
    dec.PreserveTopologyOn()
    dec.Update()
    return dec.GetOutput()


# ═══════════════════════════════════════════════════════════════════════════
# Mesh post-processing helpers
# ═══════════════════════════════════════════════════════════════════════════

def fill_holes(poly: vtk.vtkPolyData, hole_size: float = 100.0) -> vtk.vtkPolyData:
    """Fill small holes in the mesh surface."""
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(poly)
    filler.SetHoleSize(hole_size)
    filler.Update()
    return filler.GetOutput()


def smooth_mesh(poly: vtk.vtkPolyData, iterations: int = 20) -> vtk.vtkPolyData:
    """Windowed Sinc smoothing (Taubin — no shrinkage)."""
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(0.01)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def decimate_mesh(poly: vtk.vtkPolyData, reduction: float = 0.7) -> vtk.vtkPolyData:
    """Quadric decimation — *reduction* = fraction of faces to REMOVE."""
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(poly)
    decimator.SetTargetReduction(reduction)
    decimator.Update()
    return decimator.GetOutput()


def compute_normals(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Recompute consistent outward-facing normals."""
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def postprocess_mesh(
    raw: vtk.vtkPolyData,
    *,
    smooth_iters: int = 20,
    dec_ratio: float = 0.5,
    label: str = "",
) -> vtk.vtkPolyData:
    """Full chain: pre-decimate → fill holes → smooth → decimate → normals."""
    mesh = raw

    # Fast pre-decimate if huge
    if mesh.GetNumberOfCells() > FACE_PREDECIMATE_THRESHOLD:
        console.print(f"  [yellow]Pre-decimating (DecimatePro 80%) …[/yellow]")
        mesh = fast_predecimate(mesh, reduction=0.80)
        console.print(
            f"  After pre-decimate: {mesh.GetNumberOfPoints():,} pts, "
            f"{mesh.GetNumberOfCells():,} faces"
        )

    # Fill holes
    mesh = fill_holes(mesh, hole_size=100.0)

    # Smooth
    mesh = smooth_mesh(mesh, iterations=smooth_iters)

    # Final decimate
    if dec_ratio > 0:
        mesh = decimate_mesh(mesh, reduction=dec_ratio)

    # Normals
    mesh = compute_normals(mesh)
    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# OBJ writers
# ═══════════════════════════════════════════════════════════════════════════

def write_obj(poly: vtk.vtkPolyData, filepath: str) -> None:
    """Export single VTK mesh as OBJ."""
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(filepath))
    writer.SetInputData(poly)
    writer.Write()


def write_grouped_obj(
    meshes: dict[str, vtk.vtkPolyData],
    filepath: str | Path,
) -> None:
    """Write a single OBJ with named groups.

    ``meshes`` maps group name → vtkPolyData.  The resulting file::

        g bones
        v ...  f ...
        g organs
        v ...  f ...
        g body_surface
        v ...  f ...

    OBJ-aware viewers can toggle structures on/off via groups.
    """
    filepath = Path(filepath)
    vertex_offset = 0

    with open(filepath, "w") as fh:
        fh.write(f"# MedRecon Engine — human_model.obj\n")
        fh.write(f"# Groups: {', '.join(meshes.keys())}\n\n")

        for group_name, poly in meshes.items():
            fh.write(f"g {group_name}\n")

            pts = poly.GetPoints()
            n_pts = pts.GetNumberOfPoints()
            n_cells = poly.GetNumberOfCells()

            # Vertices
            for i in range(n_pts):
                x, y, z = pts.GetPoint(i)
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

            # Normals (if present)
            pn = poly.GetPointData().GetNormals()
            has_normals = pn is not None
            if has_normals:
                for i in range(n_pts):
                    nx, ny, nz = pn.GetTuple3(i)
                    fh.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

            # Faces
            for i in range(n_cells):
                cell = poly.GetCell(i)
                n_cell_pts = cell.GetNumberOfPoints()
                if n_cell_pts < 3:
                    continue
                ids = [cell.GetPointId(j) + 1 + vertex_offset
                       for j in range(n_cell_pts)]
                if has_normals:
                    face_str = " ".join(f"{v}//{v}" for v in ids)
                else:
                    face_str = " ".join(str(v) for v in ids)
                fh.write(f"f {face_str}\n")

            vertex_offset += n_pts
            fh.write("\n")


# ═══════════════════════════════════════════════════════════════════════════
# Core pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_triple_obj(
    dicom_path: str | Path,
    output_dir: str | Path,
    *,
    max_slice: float = 1.5,
    max_inplane: float = 1.0,
    smooth_iterations: int = 20,
    decimate_bone: float = 0.5,
    decimate_organ: float = 0.6,
    decimate_skin: float = 0.7,
    # Gradient organ tuning
    gauss_variance: float = 1.5,
    gradient_threshold: float = 25.0,
    hu_lower_organ: float = -200.0,
    hu_upper_organ: float = 300.0,
) -> dict:
    """DICOM → bones.obj + organs.obj + body_surface.obj + human_model.obj

    Parameters
    ----------
    dicom_path   : path to DICOM directory
    output_dir   : where to write OBJ files
    max_slice    : max allowed slice thickness (raise for 3 mm data)
    max_inplane  : max allowed in-plane pixel spacing
    smooth_iterations : windowed-sinc iterations per mesh
    decimate_bone : reduction for bones  (0.5 = remove 50 %)
    decimate_organ: reduction for organs
    decimate_skin : reduction for body_surface
    gauss_variance      : Gaussian smoothing variance for gradient organ seg (1–2)
    gradient_threshold  : gradient magnitude cutoff (20–40)
    hu_lower_organ      : lower HU bound for organ candidate mask
    hu_upper_organ      : upper HU bound for organ candidate mask
    """
    dicom_path = Path(dicom_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PrecisionConfig(
        max_allowed_slice=max_slice,
        max_allowed_inplane=max_inplane,
    )

    t_start = time.perf_counter()
    results: dict[str, dict] = {}
    final_meshes: dict[str, vtk.vtkPolyData] = {}

    # ── 1. Scan & select best CT series ───────────────────────────────
    console.print("\n[bold cyan]Step 1/9[/bold cyan] Scanning DICOM …")
    scanner = DatasetScanner()
    best = scanner.select_best_ct(dicom_path)
    console.print(f"  Series {best.series_uid[:16]}  ({len(best.file_paths)} slices)")

    # ── 2. Validate ───────────────────────────────────────────────────
    console.print("[bold cyan]Step 2/9[/bold cyan] Validating headers …")
    validator = DicomValidator(cfg)
    validator.validate(best.file_paths)
    console.print("  [green]PASS[/green]")

    # ── 3. Load volume ────────────────────────────────────────────────
    console.print("[bold cyan]Step 3/9[/bold cyan] Loading volume …")
    loader = VolumeLoader()
    volume = loader.load(best.file_paths)
    sp = volume.GetSpacing()
    sz = volume.GetSize()
    console.print(f"  Size {sz}  spacing {sp[0]:.3f}×{sp[1]:.3f}×{sp[2]:.3f} mm")

    # ── 4. HU calibration ────────────────────────────────────────────
    console.print("[bold cyan]Step 4/9[/bold cyan] Calibrating HU …")
    hu_conv = HUConverter()
    volume = hu_conv.convert(volume)

    # ── 5. HU clipping + edge-preserving diffusion ───────────────────
    console.print("[bold cyan]Step 5/9[/bold cyan] Preprocessing (clip + anisotropic diffusion) …")

    arr = sitk.GetArrayFromImage(volume).astype(np.float32)
    arr = np.clip(arr, -1000, 3000)
    clipped = sitk.GetImageFromArray(arr)
    clipped.CopyInformation(volume)
    clipped = sitk.Cast(clipped, sitk.sitkFloat32)

    diffused = sitk.CurvatureAnisotropicDiffusion(
        clipped,
        timeStep=0.04,
        conductanceParameter=3.0,
        conductanceScalingUpdateInterval=1,
        numberOfIterations=5,
    )
    diffused.CopyInformation(volume)
    volume = diffused
    console.print("  Clipped [-1000, 3000], anisotropic diffusion (5 iters)")

    # ── 6. Segmentation — three masks ────────────────────────────────
    console.print("[bold cyan]Step 6/9[/bold cyan] Segmenting …")

    # 6a. Bones — HU > 200, top-15 CC, morph close
    console.print("  [cyan]bones[/cyan]        : HU > 200 + top-15 CC + morph close")
    mask_bone = segment_bones_clean(volume, hu_threshold=200.0, max_components=15)
    nz_bone = np.count_nonzero(sitk.GetArrayFromImage(mask_bone))
    console.print(f"                 {nz_bone:,} voxels")

    # 6b. Organs — gradient + bone subtraction + skin erosion + CC filter
    console.print(
        f"  [cyan]organs[/cyan]       : gradient + subtract bones + erode skin + CC filter\n"
        f"                 (gauss={gauss_variance}, grad={gradient_threshold}, "
        f"HU [{hu_lower_organ}, {hu_upper_organ}])"
    )
    mask_organ = segment_organs_gradient(
        volume,
        gauss_variance=gauss_variance,
        gradient_threshold=gradient_threshold,
        hu_lower=hu_lower_organ,
        hu_upper=hu_upper_organ,
        bone_threshold=200.0,
        skin_erode_radius=5,
        max_organ_components=8,
    )
    nz_organ = np.count_nonzero(sitk.GetArrayFromImage(mask_organ))
    console.print(f"                 {nz_organ:,} voxels")

    # 6c. Body surface — largest CC of HU > -300, morphological close
    console.print("  [cyan]body_surface[/cyan] : largest CC of HU > -300 + morph close")
    mask_skin = segment_body_surface(volume)
    nz_skin = np.count_nonzero(sitk.GetArrayFromImage(mask_skin))
    console.print(f"                 {nz_skin:,} voxels")

    # ── 7 + 8. Marching cubes + post-process ─────────────────────────
    mesh_specs = [
        ("bones",        mask_bone,  decimate_bone),
        ("organs",       mask_organ, decimate_organ),
        ("body_surface", mask_skin,  decimate_skin),
    ]

    for name, mask, dec_ratio in mesh_specs:
        console.print(
            f"\n[bold cyan]Step 7/9[/bold cyan] Meshing "
            f"[bold yellow]{name}[/bold yellow] …"
        )
        t0 = time.perf_counter()

        nz = np.count_nonzero(sitk.GetArrayFromImage(mask))
        if nz == 0:
            console.print(f"  [red]SKIP — empty mask for {name}[/red]")
            results[name] = {"status": "EMPTY", "voxels": 0}
            continue

        # Downsample huge masks
        mesh_mask = mask
        if nz > VOXEL_DOWNSAMPLE_THRESHOLD:
            console.print(
                f"  [yellow]Large mask ({nz:,} voxels) — downsampling 2×[/yellow]"
            )
            mesh_mask = downsample_mask(mask, factor=2)
            nz_ds = np.count_nonzero(sitk.GetArrayFromImage(mesh_mask))
            console.print(f"  Downsampled: {nz_ds:,} voxels")

        # Marching cubes
        raw_mesh = generate_mesh(mesh_mask)
        raw_pts = raw_mesh.GetNumberOfPoints()
        raw_faces = raw_mesh.GetNumberOfCells()
        console.print(f"  Raw: {raw_pts:,} pts, {raw_faces:,} faces")

        # Post-process: pre-decimate → fill holes → smooth → decimate → normals
        console.print(f"[bold cyan]Step 8/9[/bold cyan] Post-processing {name} …")
        mesh = postprocess_mesh(
            raw_mesh,
            smooth_iters=smooth_iterations,
            dec_ratio=dec_ratio,
            label=name,
        )

        final_pts = mesh.GetNumberOfPoints()
        final_faces = mesh.GetNumberOfCells()
        elapsed = time.perf_counter() - t0
        console.print(
            f"  Final: {final_pts:,} pts, {final_faces:,} faces  "
            f"({elapsed:.1f}s)"
        )

        results[name] = {
            "status": "OK",
            "voxels": nz,
            "raw_faces": raw_faces,
            "final_pts": final_pts,
            "final_faces": final_faces,
            "time_s": round(elapsed, 1),
        }
        final_meshes[name] = mesh

    # ── 9. Export ─────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 9/9[/bold cyan] Exporting OBJ …")

    # 9a. Individual OBJ files
    for name, mesh in final_meshes.items():
        obj_path = output_dir / f"{name}.obj"
        write_obj(mesh, str(obj_path))
        mb = obj_path.stat().st_size / (1024 * 1024)
        console.print(f"  {obj_path.name}  →  {mb:.1f} MB")
        results[name]["file"] = str(obj_path)
        results[name]["size_mb"] = round(mb, 1)

    # 9b. Grouped OBJ — single file with g bones / g organs / g body_surface
    grouped_path = output_dir / "human_model.obj"
    console.print(f"\n  Writing grouped OBJ → [bold]{grouped_path.name}[/bold] …")
    write_grouped_obj(final_meshes, grouped_path)
    gmb = grouped_path.stat().st_size / (1024 * 1024)
    console.print(f"  human_model.obj  →  {gmb:.1f} MB")

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start

    table = Table(title="Triple OBJ Output (v4 — clean separation)")
    table.add_column("Mesh", style="cyan")
    table.add_column("Voxels", justify="right")
    table.add_column("Raw Faces", justify="right")
    table.add_column("Final Faces", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Status", style="bold")

    for name in ["bones", "organs", "body_surface"]:
        r = results.get(name, {})
        status = r.get("status", "SKIP")
        color = "green" if status == "OK" else "red"
        table.add_row(
            name,
            f"{r.get('voxels', 0):,}",
            (f"{r['raw_faces']:,}" if isinstance(r.get("raw_faces"), int) else "-"),
            (f"{r['final_faces']:,}" if isinstance(r.get("final_faces"), int) else "-"),
            str(r.get("size_mb", "-")),
            str(r.get("time_s", "-")),
            f"[{color}]{status}[/{color}]",
        )

    console.print()
    console.print(table)
    console.print(f"\n  [dim]human_model.obj  →  {gmb:.1f} MB (grouped)[/dim]")
    console.print(f"\n[bold green]Total time: {total_time:.1f}s[/bold green]\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

@click.command("triple-obj")
@click.option("--dicom", required=True, type=click.Path(exists=True),
              help="Path to DICOM directory")
@click.option("--output", default="./output", type=click.Path(),
              help="Output directory for OBJ files")
@click.option("--max-slice", default=1.5, type=float,
              help="Max allowed slice thickness (mm)")
@click.option("--max-inplane", default=1.0, type=float,
              help="Max allowed in-plane spacing (mm)")
@click.option("--smooth", default=20, type=int,
              help="Smoothing iterations (0 to skip)")
@click.option("--dec-bone", default=0.5, type=float,
              help="Decimation ratio for bones (0-1, higher=more reduction)")
@click.option("--dec-organ", default=0.6, type=float,
              help="Decimation ratio for organs")
@click.option("--dec-skin", default=0.7, type=float,
              help="Decimation ratio for body_surface")
@click.option("--gauss-var", default=1.5, type=float,
              help="Gaussian variance for organ gradient (1-2)")
@click.option("--grad-thresh", default=25.0, type=float,
              help="Gradient magnitude threshold for organ edges (20-40)")
@click.option("--hu-lower", default=-200.0, type=float,
              help="Lower HU for organ candidate window")
@click.option("--hu-upper", default=300.0, type=float,
              help="Upper HU for organ candidate window")
def cli(dicom, output, max_slice, max_inplane, smooth,
        dec_bone, dec_organ, dec_skin,
        gauss_var, grad_thresh, hu_lower, hu_upper):
    """Generate bones.obj, organs.obj, body_surface.obj + human_model.obj."""
    run_triple_obj(
        dicom_path=dicom,
        output_dir=output,
        max_slice=max_slice,
        max_inplane=max_inplane,
        smooth_iterations=smooth,
        decimate_bone=dec_bone,
        decimate_organ=dec_organ,
        decimate_skin=dec_skin,
        gauss_variance=gauss_var,
        gradient_threshold=grad_thresh,
        hu_lower_organ=hu_lower,
        hu_upper_organ=hu_upper,
    )


if __name__ == "__main__":
    cli()
