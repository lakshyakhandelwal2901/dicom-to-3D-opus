"""Quick test of the hybrid pipeline using existing TotalSegmentator labels."""
import time, sys
from pathlib import Path
import SimpleITK as sitk

# Step 1: Load existing CT + labels (skip DICOM conversion & TotalSegmentator)
nifti_path = Path("output/ai_L067_v2/scan.nii.gz")
label_dir = Path("output/ai_L067_v2/labels")
output_dir = Path("output/hybrid_L067")
output_dir.mkdir(parents=True, exist_ok=True)

t_start = time.perf_counter()
print("Loading CT volume...")
ct_image = sitk.ReadImage(str(nifti_path))
print(f"  Size: {ct_image.GetSize()}, Spacing: {ct_image.GetSpacing()}")

# Step 3: HU statistics
print("\n[3/7] Computing per-tissue HU statistics ...")
from medrecon_engine.analysis.hu_analyzer import analyze_all_labels, save_hu_profile
profiles = analyze_all_labels(ct_image, label_dir)
save_hu_profile(profiles, output_dir / "hu_profile.json", scan_id="L067")

# Step 4: Adaptive thresholds
print("\n[4/7] Deriving adaptive thresholds ...")
from medrecon_engine.analysis.threshold_generator import derive_thresholds, thresholds_summary
thresholds = derive_thresholds(profiles)
print(thresholds_summary(thresholds))

# Step 5: Classic segmentation
print("\n[5/7] Running adaptive segmentation ...")
from medrecon_engine.analysis.adaptive_segmenter import segment_all_tissues
tissue_masks = segment_all_tissues(ct_image, thresholds, label_dir=label_dir)
print(f"  Got {len(tissue_masks)} tissue masks")

# Step 6: Gradient-guided mesh extraction (with auto-downsampling)
print("\n[6/7] Generating gradient-guided meshes ...")
import numpy as np
from medrecon_engine.mesh.vtk_generator import generate_tissue_mesh
from medrecon_engine.mesh.mesh_from_labels import _organ_bbox

ct_f = sitk.Cast(ct_image, sitk.sitkFloat32)
vtk_meshes = {}

for tissue_name, mask in tissue_masks.items():
    t0 = time.perf_counter()
    arr = sitk.GetArrayFromImage(mask)
    nz = int(np.count_nonzero(arr))
    if nz == 0:
        continue

    start_idx, crop_size = _organ_bbox(arr)
    ct_crop = sitk.RegionOfInterest(ct_f, crop_size, start_idx)
    mask_crop = sitk.RegionOfInterest(mask, crop_size, start_idx)
    crop_vol = int(np.prod(crop_size))

    mesh = generate_tissue_mesh(ct_crop, mask_crop)
    if mesh is None:
        print(f"  {tissue_name}: empty mesh")
        continue

    print(f"  {tissue_name}: {mesh.GetNumberOfPoints()} pts, {mesh.GetNumberOfCells()} faces  (crop {crop_vol/1e6:.1f}M vox)  {time.perf_counter()-t0:.1f}s")
    vtk_meshes[tissue_name] = mesh

# Step 7: Export
print(f"\n[7/7] Exporting {len(vtk_meshes)} meshes ...")
from medrecon_engine.export.obj_writer import save_obj
from medrecon_engine.config.structure_groups import get_category

for name, mesh in sorted(vtk_meshes.items()):
    category = get_category(name)
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    save_obj(mesh, cat_dir / f"{name}.obj")
    print(f"  {category}/{name}.obj")

elapsed = time.perf_counter() - t_start
print(f"\n=== HYBRID PIPELINE DONE: {len(vtk_meshes)} meshes in {elapsed:.1f}s ===")
sys.stdout.flush()
