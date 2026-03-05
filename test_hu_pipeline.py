"""Quick test of the pure HU pipeline on L067."""
import sys
from pathlib import Path

dicom_path = r"D:\DATASET\Original Data\Full Dose\1mm Slice Thickness\Sharp Kernel (D45)\L067"
output_dir = "./output/hu_L067_v2"

print(f"DICOM: {dicom_path}")
print(f"Output: {output_dir}")
print()

from medrecon_engine.main import run_hu_pipeline

run_hu_pipeline(dicom_path, output_dir)
