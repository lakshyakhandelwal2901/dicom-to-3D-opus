"""
batch_run_4cases.py
====================
Run the full HU pipeline + analysis on 4 patient cases.

Cases: L067, L096, L109, L143
"""

import time
import traceback
from pathlib import Path

DATASET_ROOT = Path(r"D:\DATASET\Original Data\Full Dose\1mm Slice Thickness\Sharp Kernel (D45)")
OUTPUT_ROOT  = Path("./output")

CASES = ["L067", "L096", "L109", "L143"]


def main():
    from medrecon_engine.main import run_hu_pipeline
    from analyze_models import run_analysis

    results = {}
    t_total = time.perf_counter()

    for i, case in enumerate(CASES, 1):
        dicom_path = DATASET_ROOT / case
        output_dir = OUTPUT_ROOT / f"hu_{case}"

        print("\n" + "█" * 72)
        print(f"  CASE {i}/{len(CASES)}: {case}")
        print("█" * 72)

        if not dicom_path.exists():
            print(f"  SKIPPED — {dicom_path} not found")
            results[case] = {"status": "SKIPPED", "error": "DICOM not found"}
            continue

        try:
            # ── Step 1: Run full HU pipeline ──────────────────────────
            t0 = time.perf_counter()
            print(f"\n  ▶ Running HU pipeline …")
            run_hu_pipeline(str(dicom_path), str(output_dir))
            pipeline_time = time.perf_counter() - t0

            # ── Step 2: Run analysis ──────────────────────────────────
            print(f"\n  ▶ Running model analysis …")
            report = run_analysis(str(dicom_path), str(output_dir))

            results[case] = {
                "status": "OK",
                "pipeline_time": pipeline_time,
                "analysis_time": report.analysis_time,
                "score": report.overall_score,
                "organs_found": sum(1 for r in report.coverage if r.mesh_found),
                "organs_total": len(report.coverage),
                "missing": report.missing_structures,
                "summary": report.summary,
            }

        except Exception as exc:
            traceback.print_exc()
            results[case] = {"status": "FAILED", "error": str(exc)}

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total

    print("\n\n" + "=" * 72)
    print("  BATCH SUMMARY — 4 CASES")
    print("=" * 72)
    print(f"  Total time: {elapsed:.0f} s ({elapsed/60:.1f} min)\n")

    print(f"  {'Case':<8} {'Status':<10} {'Score':>6} {'Organs':>8} "
          f"{'Pipeline':>10} {'Analysis':>10}")
    print(f"  {'':─<8} {'':─<10} {'':─>6} {'':─>8} {'':─>10} {'':─>10}")

    for case in CASES:
        r = results[case]
        if r["status"] == "OK":
            organs = f"{r['organs_found']}/{r['organs_total']}"
            pt = f"{r['pipeline_time']:.0f}s"
            at = f"{r['analysis_time']:.0f}s"
            print(f"  {case:<8} {'OK':<10} {r['score']:>5.0f}% {organs:>8} "
                  f"{pt:>10} {at:>10}")
        else:
            print(f"  {case:<8} {r['status']:<10}")

    print()
    for case in CASES:
        r = results[case]
        if r["status"] == "OK" and r.get("summary"):
            print(f"  [{case}]")
            for s in r["summary"]:
                print(f"    • {s}")
            print()

    print("=" * 72)
    print(f"  Reports at: {OUTPUT_ROOT.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
