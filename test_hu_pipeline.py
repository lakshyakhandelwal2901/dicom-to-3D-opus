"""Quick test of the rebuilt HU pipeline on L067 with visualization."""
import sys
from pathlib import Path

dicom_path = r"D:\DATASET\Original Data\Full Dose\1mm Slice Thickness\Sharp Kernel (D45)\L067"
output_dir = "./output/hu_L067_v3"

print(f"DICOM: {dicom_path}")
print(f"Output: {output_dir}")
print()

from medrecon_engine.main import run_hu_pipeline

run_hu_pipeline(dicom_path, output_dir)

print("\nDone! Check output directory for:")
print(f"  OBJ meshes:  {output_dir}/bones/  lungs/  organs/")
print(f"  Report:       {output_dir}/segmentation_report.html")
print(f"  Viz images:   {output_dir}/visualizations/")
