"""
Batch runner — process ALL patients in D:\DATASET
==================================================
Discovers every patient folder across:
  - Original Data / Full Dose / 1mm / Sharp & Soft kernels  (10 patients × 2)
  - Original Data / Quarter Dose / 1mm / Sharp & Soft kernels (10 patients × 2)
  - Pelvic / Pelvic-Ref-001 … 058                            (58 patients)

Outputs one STL per patient into  ./output/<group>/<patient>/

Prints a summary table at the end.
"""

import sys
import time
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table

from medrecon_engine.main import run_case
from medrecon_engine.config.precision_config import PrecisionConfig

console = Console()

DATASET_ROOT = Path(r"D:\DATASET")
OUTPUT_ROOT = Path(r"./output")

# ── Discover patient directories ──────────────────────────────────────── #

def discover_patients():
    """Return list of (group_label, dicom_path) tuples."""
    patients = []

    # --- Original Data (1mm slice thickness) ---
    original = DATASET_ROOT / "Original Data"
    for dose in ["Full Dose", "Quarter Dose"]:
        for kernel_name, kernel_dir in [("Sharp", "Sharp Kernel (D45)"),
                                         ("Soft", "Soft Kernel (B30)")]:
            base = original / dose / "1mm Slice Thickness" / kernel_dir
            if not base.exists():
                continue
            for patient_dir in sorted(base.iterdir()):
                if patient_dir.is_dir():
                    group = f"Original_{dose.replace(' ', '')}_{kernel_name}"
                    patients.append((group, patient_dir.name, patient_dir))

    # --- Pelvic ---
    pelvic_base = DATASET_ROOT / "Pelvic" / "manifest-1568393181203" / "Pelvic-Reference-Data"
    if pelvic_base.exists():
        for patient_dir in sorted(pelvic_base.iterdir()):
            if patient_dir.is_dir() and patient_dir.name.startswith("Pelvic-Ref-"):
                patients.append(("Pelvic", patient_dir.name, patient_dir))

    return patients


# ── Main ──────────────────────────────────────────────────────────────── #

def main():
    patients = discover_patients()
    console.print(f"\n[bold cyan]Discovered {len(patients)} patient folders[/bold cyan]\n")

    # Config: relax slice thickness for 3mm Pelvic data
    cfg_strict = PrecisionConfig()                        # 1mm data → default strict
    cfg_relaxed = PrecisionConfig(
        max_allowed_slice=5.0,
        max_allowed_inplane=2.0,
    )

    results = []
    total = len(patients)

    for i, (group, name, path) in enumerate(patients, 1):
        console.rule(f"[bold yellow][{i}/{total}] {group} / {name}[/bold yellow]")

        cfg = cfg_relaxed if group == "Pelvic" else cfg_strict
        out_dir = OUTPUT_ROOT / group / name

        t0 = time.perf_counter()
        try:
            record = run_case(
                dicom_path=path,
                anatomy="bone",
                output_dir=out_dir,
                config=cfg,
            )
            elapsed = time.perf_counter() - t0
            results.append({
                "group": group,
                "patient": name,
                "status": "OK",
                "grade": record.confidence_grade,
                "score": f"{record.confidence_score:.3f}",
                "vertices": record.mesh_vertices,
                "faces": record.mesh_faces,
                "time_s": f"{elapsed:.1f}",
                "error": "",
            })
            console.print(f"[green]  ✓ {name} — {record.confidence_grade} ({record.confidence_score:.3f}) in {elapsed:.1f}s[/green]")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append({
                "group": group,
                "patient": name,
                "status": "FAIL",
                "grade": "-",
                "score": "-",
                "vertices": "-",
                "faces": "-",
                "time_s": f"{elapsed:.1f}",
                "error": str(exc)[:120],
            })
            console.print(f"[red]  ✗ {name} — {exc}[/red]")
            traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────── #
    console.print("\n")
    console.rule("[bold cyan]BATCH SUMMARY[/bold cyan]")

    table = Table(title=f"MedRecon Batch — {len(results)} cases")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Group", style="cyan")
    table.add_column("Patient", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Grade", style="magenta")
    table.add_column("Score")
    table.add_column("Vertices", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Error", style="red")

    ok = fail = 0
    for idx, r in enumerate(results, 1):
        style = "green" if r["status"] == "OK" else "red"
        table.add_row(
            str(idx), r["group"], r["patient"],
            f"[{style}]{r['status']}[/{style}]",
            r["grade"], r["score"],
            str(r["vertices"]), str(r["faces"]),
            r["time_s"], r["error"],
        )
        if r["status"] == "OK":
            ok += 1
        else:
            fail += 1

    console.print(table)
    console.print(f"\n[bold green]{ok} succeeded[/bold green]  |  [bold red]{fail} failed[/bold red]  |  Total: {len(results)}\n")


if __name__ == "__main__":
    main()
