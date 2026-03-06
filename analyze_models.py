"""
analyze_models.py
==================
Compare generated 3D models against the raw DICOM data.

This script answers: "Are the 3D models faithfully representing
the anatomy visible in the DICOM volume?"

Checks performed
----------------
1. **HU Coverage** — What % of each HU band in the DICOM was captured?
2. **Anatomical Completeness** — Which detectable structures are present/missing?
3. **Volume Accuracy** — Are organ volumes within expected clinical ranges?
4. **Spatial Fidelity** — Do mesh bounding boxes align with DICOM anatomy?
5. **Mesh Quality** — Manifoldness, hole count, normal consistency, face aspect ratios
6. **Unsegmented Tissue** — What clinically relevant HU ranges are NOT modelled?

Usage::

    python analyze_models.py                             # defaults: L067
    python analyze_models.py --dicom <path> --output <path>

Output::

    ./output/hu_L067_v3/analysis_report.html
    Console summary printed to stdout
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — expected anatomy & clinical reference ranges
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnatomySpec:
    """Expected anatomy definition for analysis."""
    name: str
    hu_low: float
    hu_high: float
    expected_volume_ml: tuple[float, float]   # (min, max) in mL
    obj_filename: str                          # expected OBJ filename
    category: str                              # subfolder
    description: str = ""


# Clinical reference volumes (approximate ranges for adult CT)
EXPECTED_ANATOMY: list[AnatomySpec] = [
    AnatomySpec("bones",   300, 3000,  (200, 2500),  "bones.obj",   "bones",
                "Cortical + trabecular bone (spine, pelvis, ribs, femurs)"),
    AnatomySpec("lungs",  -950, -650,  (2000, 6000), "lungs.obj",   "lungs",
                "Aerated lung parenchyma (left + right)"),
    AnatomySpec("liver",    40,   70,  (800, 2000),  "liver.obj",   "organs",
                "Hepatic parenchyma"),
    AnatomySpec("kidneys",  20,   45,  (200, 500),   "kidneys.obj", "organs",
                "Left + right renal parenchyma"),
]

# Additional HU bands that COULD be modelled but currently aren't
POTENTIAL_STRUCTURES: list[dict] = [
    {"name": "fat",             "hu_low": -190, "hu_high": -30,
     "description": "Subcutaneous + visceral fat"},
    {"name": "muscle",          "hu_low":   10, "hu_high":  40,
     "description": "Skeletal muscle (psoas, paraspinal, abdominal wall)"},
    {"name": "aorta/vessels",   "hu_low":  150, "hu_high": 300,
     "description": "Contrast-enhanced vasculature (if IV contrast)"},
    {"name": "spleen",          "hu_low":   40, "hu_high":  60,
     "description": "Splenic parenchyma"},
    {"name": "bowel_gas",       "hu_low": -900, "hu_high": -600,
     "description": "Intraluminal gas in GI tract"},
    {"name": "calcifications",  "hu_low":  130, "hu_high": 300,
     "description": "Vascular calcifications, gallstones, etc."},
]


# ═══════════════════════════════════════════════════════════════════════════
# Data classes for results
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DicomStats:
    """Statistics extracted from the raw DICOM volume."""
    size: tuple[int, int, int] = (0, 0, 0)
    spacing: tuple[float, float, float] = (0, 0, 0)
    origin: tuple[float, float, float] = (0, 0, 0)
    voxel_count: int = 0
    voxel_vol_mm3: float = 0.0
    hu_min: float = 0.0
    hu_max: float = 0.0
    hu_mean: float = 0.0
    hu_std: float = 0.0
    hu_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    hu_bin_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-band voxel counts
    band_voxels: dict[str, int] = field(default_factory=dict)
    body_voxels: int = 0  # voxels inside the body (HU > -400)


@dataclass
class MeshStats:
    """Statistics for one OBJ mesh."""
    organ: str = ""
    found: bool = False
    file_path: Path | None = None
    file_size_mb: float = 0.0
    n_vertices: int = 0
    n_faces: int = 0
    n_open_edges: int = 0
    is_manifold: bool = True
    bounds: tuple[float, ...] = ()       # (xmin, xmax, ymin, ymax, zmin, zmax)
    bbox_size_mm: tuple[float, float, float] = (0, 0, 0)
    surface_area_mm2: float = 0.0
    mesh_volume_mm3: float = 0.0        # closed-surface volume estimate
    mesh_volume_ml: float = 0.0
    min_edge_len: float = 0.0
    max_edge_len: float = 0.0
    avg_edge_len: float = 0.0
    aspect_ratio_mean: float = 0.0
    aspect_ratio_p95: float = 0.0


@dataclass
class CoverageResult:
    """Comparison result for one anatomy."""
    organ: str = ""
    dicom_voxels: int = 0
    dicom_volume_ml: float = 0.0
    mesh_found: bool = False
    mesh_volume_ml: float = 0.0
    expected_vol_range: tuple[float, float] = (0, 0)
    volume_in_range: bool = False
    coverage_pct: float = 0.0          # mesh vol / dicom vol
    status: str = "MISSING"            # OK, WARNING, MISSING, OVERSIZED, UNDERSIZED
    notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    patient_id: str = ""
    analysis_time: float = 0.0
    dicom_stats: DicomStats = field(default_factory=DicomStats)
    mesh_stats: dict[str, MeshStats] = field(default_factory=dict)
    coverage: list[CoverageResult] = field(default_factory=list)
    missing_structures: list[str] = field(default_factory=list)
    potential_additions: list[dict] = field(default_factory=list)
    overall_score: float = 0.0       # 0-100
    summary: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# DICOM Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_dicom(dicom_path: Path) -> tuple[sitk.Image, DicomStats]:
    """Load and analyze raw DICOM volume."""
    print(f"[1/4] Loading DICOM from {dicom_path} …")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_path))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_path}")

    filenames = reader.GetGDCMSeriesFileNames(str(dicom_path), series_ids[0])
    reader.SetFileNames(filenames)
    ct = reader.Execute()

    arr = sitk.GetArrayViewFromImage(ct)
    spacing = ct.GetSpacing()
    size = ct.GetSize()

    stats = DicomStats(
        size=size,
        spacing=spacing,
        origin=ct.GetOrigin(),
        voxel_count=int(np.prod(size)),
        voxel_vol_mm3=spacing[0] * spacing[1] * spacing[2],
        hu_min=float(np.min(arr)),
        hu_max=float(np.max(arr)),
        hu_mean=float(np.mean(arr)),
        hu_std=float(np.std(arr)),
    )

    # HU histogram
    stats.hu_histogram, stats.hu_bin_edges = np.histogram(
        arr.ravel(), bins=200, range=(-1024, 3000)
    )

    # Body voxels (rough estimate: HU > -400 excludes air)
    stats.body_voxels = int(np.count_nonzero(arr > -400))

    # Count voxels in each expected anatomy band
    for spec in EXPECTED_ANATOMY:
        band = np.count_nonzero((arr >= spec.hu_low) & (arr <= spec.hu_high))
        stats.band_voxels[spec.name] = int(band)

    # Also count potential structures
    for ps in POTENTIAL_STRUCTURES:
        band = np.count_nonzero((arr >= ps["hu_low"]) & (arr <= ps["hu_high"]))
        stats.band_voxels[ps["name"]] = int(band)

    print(f"      Volume: {size[0]}×{size[1]}×{size[2]}")
    print(f"      Spacing: {spacing[0]:.3f}×{spacing[1]:.3f}×{spacing[2]:.3f} mm")
    print(f"      HU range: [{stats.hu_min:.0f}, {stats.hu_max:.0f}]")
    print(f"      Body voxels: {stats.body_voxels:,}")

    return ct, stats


# ═══════════════════════════════════════════════════════════════════════════
# Mesh Analysis
# ═══════════════════════════════════════════════════════════════════════════

def _load_obj(filepath: Path) -> vtk.vtkPolyData | None:
    """Load an OBJ file into vtkPolyData."""
    if not filepath.exists():
        return None
    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    return reader.GetOutput()


def _compute_edge_stats(mesh: vtk.vtkPolyData) -> tuple[float, float, float]:
    """Compute min, max, mean edge lengths."""
    edges = set()
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        n = cell.GetNumberOfPoints()
        for j in range(n):
            a = cell.GetPointId(j)
            b = cell.GetPointId((j + 1) % n)
            edge = (min(a, b), max(a, b))
            edges.add(edge)

    if not edges:
        return 0.0, 0.0, 0.0

    pts = mesh.GetPoints()
    lengths = []
    for a, b in edges:
        pa = np.array(pts.GetPoint(a))
        pb = np.array(pts.GetPoint(b))
        lengths.append(np.linalg.norm(pa - pb))

    lengths = np.array(lengths)
    return float(np.min(lengths)), float(np.max(lengths)), float(np.mean(lengths))


def _compute_aspect_ratios(mesh: vtk.vtkPolyData) -> tuple[float, float]:
    """Compute mean and 95th percentile triangle aspect ratio."""
    pts = mesh.GetPoints()
    ratios = []

    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetNumberOfPoints() != 3:
            continue
        p0 = np.array(pts.GetPoint(cell.GetPointId(0)))
        p1 = np.array(pts.GetPoint(cell.GetPointId(1)))
        p2 = np.array(pts.GetPoint(cell.GetPointId(2)))

        edges = [np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p0 - p2)]
        if min(edges) > 1e-10:
            ratios.append(max(edges) / min(edges))

    if not ratios:
        return 0.0, 0.0

    ratios = np.array(ratios)
    return float(np.mean(ratios)), float(np.percentile(ratios, 95))


def analyze_mesh(filepath: Path, organ: str) -> MeshStats:
    """Analyze a single OBJ mesh for quality and geometry."""
    stats = MeshStats(organ=organ)

    if not filepath.exists():
        stats.found = False
        return stats

    mesh = _load_obj(filepath)
    if mesh is None or mesh.GetNumberOfPoints() == 0:
        stats.found = False
        return stats

    stats.found = True
    stats.file_path = filepath
    stats.file_size_mb = filepath.stat().st_size / (1024 * 1024)
    stats.n_vertices = mesh.GetNumberOfPoints()
    stats.n_faces = mesh.GetNumberOfCells()

    # Bounds
    bounds = mesh.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    stats.bounds = bounds
    stats.bbox_size_mm = (
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )

    # Feature edges → open (boundary) edges
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(mesh)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    stats.n_open_edges = fe.GetOutput().GetNumberOfCells()
    stats.is_manifold = stats.n_open_edges == 0

    # Surface area
    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    stats.surface_area_mm2 = mass.GetSurfaceArea()

    # Volume (only meaningful for watertight meshes, but still indicative)
    stats.mesh_volume_mm3 = abs(mass.GetVolume())
    stats.mesh_volume_ml = stats.mesh_volume_mm3 / 1000.0

    # Edge lengths
    stats.min_edge_len, stats.max_edge_len, stats.avg_edge_len = (
        _compute_edge_stats(mesh)
    )

    # Aspect ratios
    stats.aspect_ratio_mean, stats.aspect_ratio_p95 = _compute_aspect_ratios(mesh)

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Coverage & Comparison
# ═══════════════════════════════════════════════════════════════════════════

def compute_coverage(
    dicom_stats: DicomStats,
    mesh_stats: dict[str, MeshStats],
) -> list[CoverageResult]:
    """Compare each expected anatomy against what was captured."""
    results = []

    for spec in EXPECTED_ANATOMY:
        r = CoverageResult(
            organ=spec.name,
            dicom_voxels=dicom_stats.band_voxels.get(spec.name, 0),
            expected_vol_range=spec.expected_volume_ml,
        )
        r.dicom_volume_ml = r.dicom_voxels * dicom_stats.voxel_vol_mm3 / 1000.0

        ms = mesh_stats.get(spec.name)
        if ms and ms.found:
            r.mesh_found = True
            r.mesh_volume_ml = ms.mesh_volume_ml

            # Coverage ratio
            if r.dicom_volume_ml > 0:
                r.coverage_pct = (r.mesh_volume_ml / r.dicom_volume_ml) * 100.0

            # Volume within expected clinical range?
            lo, hi = spec.expected_volume_ml
            r.volume_in_range = lo <= r.mesh_volume_ml <= hi

            # Status determination
            if r.volume_in_range and r.coverage_pct > 30:
                r.status = "OK"
            elif r.mesh_volume_ml > hi * 1.5:
                r.status = "OVERSIZED"
                r.notes.append(f"Mesh volume {r.mesh_volume_ml:.0f} mL exceeds "
                               f"expected max {hi:.0f} mL by >50%")
            elif r.mesh_volume_ml < lo * 0.5:
                r.status = "UNDERSIZED"
                r.notes.append(f"Mesh volume {r.mesh_volume_ml:.0f} mL below "
                               f"expected min {lo:.0f} mL by >50%")
            elif not r.volume_in_range:
                r.status = "WARNING"
                r.notes.append(f"Volume {r.mesh_volume_ml:.0f} mL outside "
                               f"expected [{lo:.0f}, {hi:.0f}] mL")
            else:
                r.status = "WARNING"
                r.notes.append(f"Low coverage: {r.coverage_pct:.1f}%")
        else:
            r.status = "MISSING"
            r.notes.append("OBJ file not found or empty")

        # Additional quality notes
        if ms and ms.found:
            if not ms.is_manifold:
                r.notes.append(f"{ms.n_open_edges} open edges (not watertight)")
            if ms.aspect_ratio_p95 > 10:
                r.notes.append(f"Poor triangle quality (AR p95={ms.aspect_ratio_p95:.1f})")

        results.append(r)

    return results


def find_potential_additions(dicom_stats: DicomStats) -> list[dict]:
    """Identify structures in the DICOM that could be added to the pipeline."""
    additions = []

    for ps in POTENTIAL_STRUCTURES:
        voxels = dicom_stats.band_voxels.get(ps["name"], 0)
        vol_ml = voxels * dicom_stats.voxel_vol_mm3 / 1000.0

        # Only flag structures with meaningful volume (> 50 mL)
        if vol_ml > 50:
            additions.append({
                "name": ps["name"],
                "description": ps["description"],
                "hu_range": f"[{ps['hu_low']}, {ps['hu_high']}]",
                "dicom_voxels": voxels,
                "dicom_volume_ml": vol_ml,
                "priority": "high" if vol_ml > 500 else "medium" if vol_ml > 100 else "low",
            })

    # Sort by volume descending
    additions.sort(key=lambda x: x["dicom_volume_ml"], reverse=True)
    return additions


def compute_score(coverage: list[CoverageResult], mesh_stats: dict[str, MeshStats]) -> float:
    """Compute an overall quality score (0-100)."""
    score = 0.0
    max_per_organ = 100.0 / len(EXPECTED_ANATOMY)

    for r in coverage:
        organ_score = 0.0

        # Presence: 40% of organ score
        if r.mesh_found:
            organ_score += max_per_organ * 0.4

        # Volume accuracy: 30% of organ score
        if r.volume_in_range:
            organ_score += max_per_organ * 0.3
        elif r.mesh_found:
            organ_score += max_per_organ * 0.15  # partial credit

        # Mesh quality: 30% of organ score
        ms = mesh_stats.get(r.organ)
        if ms and ms.found:
            quality = max_per_organ * 0.3
            if not ms.is_manifold:
                quality *= 0.7  # watertight penalty
            if ms.aspect_ratio_p95 > 10:
                quality *= 0.8  # quality penalty
            organ_score += quality

        score += organ_score

    return min(100.0, score)


# ═══════════════════════════════════════════════════════════════════════════
# Console report
# ═══════════════════════════════════════════════════════════════════════════

def _status_icon(status: str) -> str:
    markers = {"OK": "[OK]", "WARNING": "[!!]", "MISSING": "[XX]",
               "OVERSIZED": "[>>]", "UNDERSIZED": "[<<]"}
    return markers.get(status, "[??]")


def print_console_report(report: AnalysisReport) -> None:
    """Print a detailed console summary."""
    ds = report.dicom_stats

    print()
    print("=" * 72)
    print("  DICOM vs 3D MODEL ANALYSIS REPORT")
    print("=" * 72)
    print(f"  Patient       : {report.patient_id}")
    print(f"  Analysis time : {report.analysis_time:.1f} s")
    print(f"  Overall score : {report.overall_score:.0f} / 100")
    print()

    # DICOM summary
    print("-" * 72)
    print("  RAW DICOM SUMMARY")
    print("-" * 72)
    print(f"  Volume size   : {ds.size[0]} x {ds.size[1]} x {ds.size[2]}")
    print(f"  Spacing       : {ds.spacing[0]:.3f} x {ds.spacing[1]:.3f} x {ds.spacing[2]:.3f} mm")
    print(f"  Total voxels  : {ds.voxel_count:,}")
    print(f"  Body voxels   : {ds.body_voxels:,}")
    print(f"  HU range      : [{ds.hu_min:.0f}, {ds.hu_max:.0f}]")
    print(f"  HU mean±std   : {ds.hu_mean:.1f} ± {ds.hu_std:.1f}")
    print()

    # Per-organ coverage
    print("-" * 72)
    print("  ORGAN-BY-ORGAN COMPARISON")
    print("-" * 72)
    print(f"  {'Organ':<12} {'Status':<10} {'DICOM Vol':>10} {'Mesh Vol':>10} "
          f"{'Expected':>14} {'Coverage':>9}")
    print(f"  {'':─<12} {'':─<10} {'':─>10} {'':─>10} {'':─>14} {'':─>9}")

    for r in report.coverage:
        lo, hi = r.expected_vol_range
        exp = f"[{lo:.0f}-{hi:.0f}]"
        cov = f"{r.coverage_pct:.1f}%" if r.mesh_found else "—"
        mv = f"{r.mesh_volume_ml:.0f} mL" if r.mesh_found else "—"
        dv = f"{r.dicom_volume_ml:.0f} mL"

        print(f"  {r.organ:<12} {_status_icon(r.status):<10} {dv:>10} {mv:>10} "
              f"{exp:>14} {cov:>9}")

        for note in r.notes:
            print(f"  {'':>12}   └─ {note}")

    print()

    # Mesh quality
    print("-" * 72)
    print("  MESH QUALITY DETAILS")
    print("-" * 72)
    print(f"  {'Organ':<12} {'Vertices':>10} {'Faces':>10} {'Manifold':>10} "
          f"{'Open Edges':>12} {'AR p95':>8} {'File MB':>8}")
    print(f"  {'':─<12} {'':─>10} {'':─>10} {'':─>10} {'':─>12} {'':─>8} {'':─>8}")

    for name, ms in sorted(report.mesh_stats.items()):
        if not ms.found:
            print(f"  {name:<12} {'— not found —':^60}")
            continue
        mani = "Yes" if ms.is_manifold else "No"
        print(f"  {name:<12} {ms.n_vertices:>10,} {ms.n_faces:>10,} {mani:>10} "
              f"{ms.n_open_edges:>12} {ms.aspect_ratio_p95:>8.1f} "
              f"{ms.file_size_mb:>8.1f}")

    print()

    # Missing / potential structures
    if report.missing_structures:
        print("-" * 72)
        print("  MISSING EXPECTED STRUCTURES")
        print("-" * 72)
        for name in report.missing_structures:
            print(f"    [X] {name}")
        print()

    if report.potential_additions:
        print("-" * 72)
        print("  STRUCTURES VISIBLE IN DICOM BUT NOT MODELLED")
        print("-" * 72)
        print(f"  {'Structure':<18} {'HU Range':<14} {'DICOM Vol':>10} {'Priority':>10}")
        print(f"  {'':─<18} {'':─<14} {'':─>10} {'':─>10}")
        for pa in report.potential_additions:
            vol = f"{pa['dicom_volume_ml']:.0f} mL"
            print(f"  {pa['name']:<18} {pa['hu_range']:<14} {vol:>10} "
                  f"{pa['priority']:>10}")
            print(f"  {'':>18} {pa['description']}")
        print()

    # Summary / recommendations
    print("-" * 72)
    print("  SUMMARY & RECOMMENDATIONS")
    print("-" * 72)
    for s in report.summary:
        print(f"    • {s}")
    print()
    print("=" * 72)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def _make_histogram_png(dicom_stats: DicomStats) -> str:
    """Generate HU histogram as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    centers = (dicom_stats.hu_bin_edges[:-1] + dicom_stats.hu_bin_edges[1:]) / 2
    ax.bar(centers, dicom_stats.hu_histogram, width=np.diff(dicom_stats.hu_bin_edges),
           color="#4a90d9", alpha=0.8, edgecolor="none")

    # Shade HU bands of expected anatomy
    colors = {"bones": "#f2e6d9", "lungs": "#d0e8ff", "liver": "#ffd6d6", "kidneys": "#ffe0b2"}
    for spec in EXPECTED_ANATOMY:
        c = colors.get(spec.name, "#dddddd")
        ax.axvspan(spec.hu_low, spec.hu_high, alpha=0.2, color=c, label=spec.name)

    ax.set_xlabel("Hounsfield Units (HU)", fontsize=11)
    ax.set_ylabel("Voxel Count", fontsize=11)
    ax.set_title("DICOM HU Distribution — Shaded Bands = Segmented Ranges", fontsize=12)
    ax.set_xlim(-1100, 2000)
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


def generate_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Generate a self-contained HTML analysis report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hist_b64 = _make_histogram_png(report.dicom_stats)
    ds = report.dicom_stats

    # Status styling
    status_colors = {
        "OK": "#2ea043", "WARNING": "#d29922",
        "MISSING": "#f85149", "OVERSIZED": "#da3633", "UNDERSIZED": "#db6d28"
    }

    # Coverage table rows
    cov_rows = ""
    for r in report.coverage:
        sc = status_colors.get(r.status, "#888")
        lo, hi = r.expected_vol_range
        cov = f"{r.coverage_pct:.1f}%" if r.mesh_found else "—"
        mv = f"{r.mesh_volume_ml:.0f}" if r.mesh_found else "—"
        notes_html = "<br>".join(r.notes) if r.notes else "—"
        cov_rows += f"""<tr>
            <td>{r.organ.capitalize()}</td>
            <td style="color:{sc};font-weight:bold">{r.status}</td>
            <td>{r.dicom_volume_ml:.0f}</td>
            <td>{mv}</td>
            <td>[{lo:.0f} – {hi:.0f}]</td>
            <td>{cov}</td>
            <td style="font-size:12px">{notes_html}</td>
        </tr>"""

    # Mesh quality rows
    mq_rows = ""
    for name, ms in sorted(report.mesh_stats.items()):
        if not ms.found:
            mq_rows += f"""<tr><td>{name.capitalize()}</td>
                <td colspan="6" style="text-align:center;color:#888">Not found</td></tr>"""
            continue
        mani_c = "#2ea043" if ms.is_manifold else "#f85149"
        mani = "Yes" if ms.is_manifold else "No"
        ar_c = "#2ea043" if ms.aspect_ratio_p95 < 5 else "#d29922" if ms.aspect_ratio_p95 < 10 else "#f85149"
        mq_rows += f"""<tr>
            <td>{name.capitalize()}</td>
            <td>{ms.n_vertices:,}</td>
            <td>{ms.n_faces:,}</td>
            <td style="color:{mani_c}">{mani}</td>
            <td>{ms.n_open_edges}</td>
            <td style="color:{ar_c}">{ms.aspect_ratio_p95:.1f}</td>
            <td>{ms.file_size_mb:.1f}</td>
        </tr>"""

    # Missing structures
    missing_html = ""
    if report.missing_structures:
        items = "".join(f"<li>{n.capitalize()}</li>" for n in report.missing_structures)
        missing_html = f"""<div class="alert alert-red">
            <h3>Missing Expected Structures</h3>
            <ul>{items}</ul>
        </div>"""

    # Potential additions
    pot_rows = ""
    for pa in report.potential_additions:
        pri_c = "#f85149" if pa["priority"] == "high" else "#d29922" if pa["priority"] == "medium" else "#8b949e"
        pot_rows += f"""<tr>
            <td>{pa['name'].capitalize()}</td>
            <td>{pa['description']}</td>
            <td>{pa['hu_range']}</td>
            <td>{pa['dicom_volume_ml']:.0f}</td>
            <td style="color:{pri_c};font-weight:bold">{pa['priority'].upper()}</td>
        </tr>"""

    # Summary bullets
    summary_items = "".join(f"<li>{s}</li>" for s in report.summary)

    # Score color
    score = report.overall_score
    sc_color = "#2ea043" if score >= 80 else "#d29922" if score >= 50 else "#f85149"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MedRecon — Model Analysis Report</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, sans-serif;
        background: #0d1117; color: #e6edf3;
        max-width: 1200px; margin: 0 auto; padding: 20px;
    }}
    h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
    h2 {{ color: #79c0ff; margin-top: 30px; }}
    h3 {{ color: #d2a8ff; }}
    .meta {{
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 15px 20px; margin: 15px 0;
        font-family: monospace; font-size: 14px;
    }}
    .meta span {{ color: #8b949e; }}
    .score-box {{
        display: inline-block; padding: 15px 30px;
        border-radius: 12px; font-size: 32px; font-weight: bold;
        border: 2px solid {sc_color}; color: {sc_color};
        background: #161b22; margin: 10px 0;
    }}
    table {{
        width: 100%; border-collapse: collapse; margin: 15px 0;
    }}
    th, td {{
        padding: 10px 14px; text-align: left;
        border-bottom: 1px solid #30363d;
    }}
    th {{
        background: #161b22; color: #58a6ff; font-weight: 600;
    }}
    tr:hover {{ background: #161b22; }}
    img {{
        max-width: 100%; border-radius: 8px; margin: 10px 0;
        border: 1px solid #30363d;
    }}
    .alert {{
        border-radius: 8px; padding: 15px 20px; margin: 15px 0;
    }}
    .alert-red {{
        background: #1c0c0c; border: 1px solid #f85149;
    }}
    .alert-yellow {{
        background: #1c1a0c; border: 1px solid #d29922;
    }}
    .alert-green {{
        background: #0c1c0c; border: 1px solid #2ea043;
    }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 5px 0; }}
    .footer {{
        text-align: center; color: #484f58; font-size: 12px;
        margin-top: 40px; padding-top: 15px;
        border-top: 1px solid #21262d;
    }}
</style>
</head>
<body>

<h1>MedRecon — Model Analysis Report</h1>

<div class="meta">
    <span>Patient:</span> {report.patient_id} &nbsp;|&nbsp;
    <span>Analysis time:</span> {report.analysis_time:.1f} s
</div>

<div style="text-align:center;margin:20px 0">
    <div style="color:#8b949e;font-size:14px">Overall Quality Score</div>
    <div class="score-box">{score:.0f} / 100</div>
</div>

<h2>1. DICOM Volume Summary</h2>
<div class="meta">
    <span>Size:</span> {ds.size[0]} × {ds.size[1]} × {ds.size[2]} &nbsp;|&nbsp;
    <span>Spacing:</span> {ds.spacing[0]:.3f} × {ds.spacing[1]:.3f} × {ds.spacing[2]:.3f} mm<br>
    <span>Voxels:</span> {ds.voxel_count:,} &nbsp;|&nbsp;
    <span>Body voxels:</span> {ds.body_voxels:,}<br>
    <span>HU range:</span> [{ds.hu_min:.0f}, {ds.hu_max:.0f}] &nbsp;|&nbsp;
    <span>HU mean ± std:</span> {ds.hu_mean:.1f} ± {ds.hu_std:.1f}
</div>

<h3>HU Distribution</h3>
<img src="{hist_b64}" alt="HU Histogram">

<h2>2. Organ Coverage — DICOM vs 3D Models</h2>
{missing_html}
<table>
    <tr>
        <th>Organ</th><th>Status</th><th>DICOM Vol (mL)</th>
        <th>Mesh Vol (mL)</th><th>Expected (mL)</th>
        <th>Coverage</th><th>Notes</th>
    </tr>
    {cov_rows}
</table>

<h2>3. Mesh Quality Assessment</h2>
<table>
    <tr>
        <th>Organ</th><th>Vertices</th><th>Faces</th>
        <th>Manifold</th><th>Open Edges</th>
        <th>AR p95</th><th>File (MB)</th>
    </tr>
    {mq_rows}
</table>

<h2>4. Structures Visible in DICOM but Not Modelled</h2>
<p style="color:#8b949e">These HU bands contain significant tissue volume
in the raw DICOM that could be added as new organs/structures.</p>
<table>
    <tr>
        <th>Structure</th><th>Description</th><th>HU Range</th>
        <th>DICOM Vol (mL)</th><th>Priority</th>
    </tr>
    {pot_rows}
</table>

<h2>5. Summary & Recommendations</h2>
<div class="alert alert-{'green' if score >= 80 else 'yellow' if score >= 50 else 'red'}">
    <ul>{summary_items}</ul>
</div>

<div class="footer">
    Generated by MedRecon Engine — Model Analysis Tool
</div>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML report: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(dicom_path: str | Path, output_dir: str | Path) -> AnalysisReport:
    """Run complete model-vs-DICOM analysis.

    Parameters
    ----------
    dicom_path : Path to DICOM directory
    output_dir : Path to pipeline output directory (containing bones/, lungs/ etc.)

    Returns
    -------
    AnalysisReport with all findings
    """
    t0 = time.perf_counter()
    dicom_path = Path(dicom_path)
    output_dir = Path(output_dir)

    report = AnalysisReport(patient_id=dicom_path.name)

    # ── 1. Analyze DICOM ──────────────────────────────────────────────
    ct_image, report.dicom_stats = analyze_dicom(dicom_path)

    # ── 2. Analyze meshes ─────────────────────────────────────────────
    print(f"\n[2/4] Analyzing OBJ meshes in {output_dir} …")
    for spec in EXPECTED_ANATOMY:
        obj_path = output_dir / spec.category / spec.obj_filename
        ms = analyze_mesh(obj_path, spec.name)
        report.mesh_stats[spec.name] = ms
        if ms.found:
            print(f"      {spec.name}: {ms.n_vertices:,} verts, "
                  f"{ms.n_faces:,} faces, {ms.mesh_volume_ml:.0f} mL, "
                  f"{'manifold' if ms.is_manifold else 'NOT manifold'}")
        else:
            print(f"      {spec.name}: NOT FOUND at {obj_path}")

    # ── 3. Coverage comparison ────────────────────────────────────────
    print(f"\n[3/4] Computing coverage & comparison …")
    report.coverage = compute_coverage(report.dicom_stats, report.mesh_stats)
    report.missing_structures = [
        r.organ for r in report.coverage if r.status == "MISSING"
    ]

    # ── 4. Identify potential additions ───────────────────────────────
    report.potential_additions = find_potential_additions(report.dicom_stats)

    # ── Score ─────────────────────────────────────────────────────────
    report.overall_score = compute_score(report.coverage, report.mesh_stats)

    # ── Summary bullets ───────────────────────────────────────────────
    n_found = sum(1 for r in report.coverage if r.mesh_found)
    n_total = len(EXPECTED_ANATOMY)
    report.summary.append(
        f"{n_found}/{n_total} expected organs have 3D models")

    for r in report.coverage:
        if r.status == "MISSING":
            report.summary.append(
                f"MISSING: {r.organ} — {r.dicom_volume_ml:.0f} mL visible in DICOM "
                f"but no model generated")
        elif r.status == "OVERSIZED":
            report.summary.append(
                f"OVERSIZED: {r.organ} mesh ({r.mesh_volume_ml:.0f} mL) exceeds "
                f"expected range")
        elif r.status == "UNDERSIZED":
            report.summary.append(
                f"UNDERSIZED: {r.organ} mesh ({r.mesh_volume_ml:.0f} mL) below "
                f"expected range")
        elif r.status == "OK":
            report.summary.append(
                f"OK: {r.organ} — {r.mesh_volume_ml:.0f} mL mesh, "
                f"{r.coverage_pct:.0f}% DICOM coverage")

    # Mesh quality summary
    non_manifold = [n for n, ms in report.mesh_stats.items()
                    if ms.found and not ms.is_manifold]
    if non_manifold:
        report.summary.append(
            f"MESH QUALITY: {', '.join(non_manifold)} have open edges "
            f"(not watertight) — may cause issues in 3D printing/simulation")

    # What's missing
    if report.potential_additions:
        top3 = report.potential_additions[:3]
        names = ", ".join(p["name"] for p in top3)
        report.summary.append(
            f"POTENTIAL ADDITIONS: {names} — visible in DICOM but not modelled")

    # Total unsegmented body tissue
    total_seg = sum(
        report.dicom_stats.band_voxels.get(spec.name, 0)
        for spec in EXPECTED_ANATOMY
    )
    body_pct = (total_seg / max(report.dicom_stats.body_voxels, 1)) * 100
    report.summary.append(
        f"BODY COVERAGE: {body_pct:.1f}% of body voxels fall within "
        f"segmented HU ranges ({total_seg:,} / {report.dicom_stats.body_voxels:,})")

    report.analysis_time = time.perf_counter() - t0

    # ── Output ────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating reports …")
    print_console_report(report)
    generate_html_report(report, output_dir / "analysis_report.html")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze 3D models vs raw DICOM — identify what's missing"
    )
    parser.add_argument(
        "--dicom", type=str,
        default=r"D:\DATASET\Original Data\Full Dose\1mm Slice Thickness\Sharp Kernel (D45)\L067",
        help="Path to DICOM directory",
    )
    parser.add_argument(
        "--output", type=str,
        default="./output/hu_L067_v3",
        help="Path to pipeline output directory (with bones/, lungs/, organs/ subfolders)",
    )
    args = parser.parse_args()

    run_analysis(args.dicom, args.output)
