"""
medrecon_engine.analysis.medical_findings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated CT scan medical findings report.

Reads raw DICOM data and generates a comprehensive clinical-style
report with:
  - Patient demographics & study information
  - Scan protocol & technical parameters
  - Tissue composition analysis (HU histogram)
  - Organ-specific findings (normal vs abnormal)
  - Detected pathologies & potential risks
  - Quantitative measurements

**Disclaimer**: This is an automated computational analysis,
NOT a radiologist's interpretation. All findings require
validation by a qualified medical professional.

Usage::

    from medrecon_engine.analysis.medical_findings import generate_medical_report
    report_path = generate_medical_report(dicom_dir, output_dir)
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PatientInfo:
    """Extracted DICOM patient demographic & study metadata."""
    patient_id: str = "Unknown"
    patient_name: str = "Unknown"
    patient_age: str = "Unknown"
    patient_sex: str = "Unknown"
    patient_weight: str = "Unknown"
    study_date: str = "Unknown"
    study_description: str = "Unknown"
    series_description: str = "Unknown"
    institution: str = "Unknown"
    manufacturer: str = "Unknown"
    model_name: str = "Unknown"
    body_part: str = "Unknown"
    protocol: str = "Unknown"
    kvp: str = "Unknown"
    tube_current: str = "Unknown"
    exposure: str = "Unknown"
    slice_thickness: str = "Unknown"
    convolution_kernel: str = "Unknown"
    contrast_agent: str = "None detected"
    accession_number: str = "Unknown"
    referring_physician: str = "Unknown"


@dataclass
class TissueStats:
    """Quantitative tissue composition from HU histogram analysis."""
    total_body_voxels: int = 0
    voxel_volume_mm3: float = 0.0
    # Tissue volumes in mL
    air_volume_ml: float = 0.0           # < -900 HU
    lung_volume_ml: float = 0.0          # -950 to -650 HU
    fat_volume_ml: float = 0.0           # -190 to -30 HU
    water_fluid_volume_ml: float = 0.0   # -10 to 15 HU
    soft_tissue_volume_ml: float = 0.0   # 20 to 80 HU
    muscle_volume_ml: float = 0.0        # 10 to 40 HU
    liver_tissue_volume_ml: float = 0.0  # 40 to 70 HU
    bone_volume_ml: float = 0.0          # > 300 HU
    dense_bone_volume_ml: float = 0.0    # > 700 HU
    calcification_volume_ml: float = 0.0  # 130 to 300 HU
    contrast_volume_ml: float = 0.0      # 150 to 500 HU


@dataclass
class Finding:
    """A single medical finding with severity and detail."""
    category: str    # "normal", "abnormal", "risk", "incidental"
    organ: str       # e.g. "Lungs", "Liver", "Bones"
    title: str       # Short heading
    detail: str      # Detailed description
    severity: str    # "normal", "mild", "moderate", "severe", "critical"
    measurement: str = ""  # Optional quantitative measurement


@dataclass
class MedicalReport:
    """Complete medical findings report."""
    patient: PatientInfo = field(default_factory=PatientInfo)
    tissue: TissueStats = field(default_factory=TissueStats)
    findings: list[Finding] = field(default_factory=list)
    scan_quality: str = "Adequate"
    scan_coverage: str = ""
    volume_size: tuple[int, int, int] = (0, 0, 0)
    spacing: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hu_min: float = 0.0
    hu_max: float = 0.0
    hu_mean: float = 0.0
    hu_std: float = 0.0
    analysis_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Core analysis functions
# ═══════════════════════════════════════════════════════════════════════════

def _extract_patient_info(dicom_dir: Path) -> PatientInfo:
    """Read DICOM headers to extract patient & study metadata."""
    info = PatientInfo()

    # Find the first readable DICOM file
    dcm_files = sorted(dicom_dir.glob("*"))
    ds = None
    for f in dcm_files:
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if hasattr(ds, "Modality"):
                    break
            except Exception:
                continue

    if ds is None:
        _log.warning("Could not read any DICOM headers from %s", dicom_dir)
        return info

    # Patient demographics
    info.patient_id = str(getattr(ds, "PatientID", "Unknown"))
    raw_name = getattr(ds, "PatientName", "Unknown")
    info.patient_name = str(raw_name).replace("^", " ") if raw_name else "Unknown"
    info.patient_age = str(getattr(ds, "PatientAge", "Unknown"))
    info.patient_sex = str(getattr(ds, "PatientSex", "Unknown"))
    weight = getattr(ds, "PatientWeight", None)
    info.patient_weight = f"{float(weight):.1f} kg" if weight else "Not recorded"

    # Study information
    date = str(getattr(ds, "StudyDate", "Unknown"))
    if date != "Unknown" and len(date) == 8:
        info.study_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    else:
        info.study_date = date
    info.study_description = str(getattr(ds, "StudyDescription", "Not specified"))
    info.series_description = str(getattr(ds, "SeriesDescription", "Not specified"))
    info.institution = str(getattr(ds, "InstitutionName", "Not specified"))
    info.accession_number = str(getattr(ds, "AccessionNumber", "Not specified"))
    info.referring_physician = str(getattr(ds, "ReferringPhysicianName", "Not specified"))

    # Scanner details
    info.manufacturer = str(getattr(ds, "Manufacturer", "Unknown"))
    info.model_name = str(getattr(ds, "ManufacturerModelName", "Unknown"))

    # Protocol & acquisition
    info.body_part = str(getattr(ds, "BodyPartExamined", "Unknown"))
    info.protocol = str(getattr(ds, "ProtocolName", "Not specified"))
    kvp = getattr(ds, "KVP", None)
    info.kvp = f"{float(kvp):.0f} kVp" if kvp else "Not recorded"
    tube_ma = getattr(ds, "XRayTubeCurrent", None)
    info.tube_current = f"{float(tube_ma):.0f} mA" if tube_ma else "Not recorded"
    exposure = getattr(ds, "Exposure", None)
    info.exposure = f"{float(exposure):.0f} mAs" if exposure else "Not recorded"
    st = getattr(ds, "SliceThickness", None)
    info.slice_thickness = f"{float(st):.2f} mm" if st else "Not recorded"
    info.convolution_kernel = str(getattr(ds, "ConvolutionKernel", "Not specified"))

    # Contrast detection
    contrast = getattr(ds, "ContrastBolusAgent", None)
    if contrast and str(contrast).strip():
        info.contrast_agent = str(contrast)
    else:
        # Check ContrastBolusSequence
        seq = getattr(ds, "ContrastBolusAgentSequence", None)
        if seq:
            info.contrast_agent = "IV contrast (sequence present)"

    return info


def _compute_tissue_stats(ct_arr: np.ndarray, voxel_vol_mm3: float) -> TissueStats:
    """Compute tissue composition by analysing HU histogram."""
    stats = TissueStats()
    stats.voxel_volume_mm3 = voxel_vol_mm3

    # Body mask: exclude external air (< -500 HU and at volume boundary)
    body_mask = ct_arr > -500
    stats.total_body_voxels = int(np.count_nonzero(body_mask))

    def _volume(mask: np.ndarray) -> float:
        return int(np.count_nonzero(mask)) * voxel_vol_mm3 / 1000.0

    # Tissue classification by HU ranges
    stats.air_volume_ml = _volume(ct_arr < -900)
    stats.lung_volume_ml = _volume((ct_arr >= -950) & (ct_arr <= -650))
    stats.fat_volume_ml = _volume((ct_arr >= -190) & (ct_arr <= -30))
    stats.water_fluid_volume_ml = _volume((ct_arr >= -10) & (ct_arr <= 15))
    stats.soft_tissue_volume_ml = _volume((ct_arr >= 20) & (ct_arr <= 80))
    stats.muscle_volume_ml = _volume((ct_arr >= 10) & (ct_arr <= 40))
    stats.liver_tissue_volume_ml = _volume((ct_arr >= 40) & (ct_arr <= 70))
    stats.bone_volume_ml = _volume(ct_arr > 300)
    stats.dense_bone_volume_ml = _volume(ct_arr > 700)
    stats.calcification_volume_ml = _volume((ct_arr >= 130) & (ct_arr <= 300))
    stats.contrast_volume_ml = _volume((ct_arr >= 150) & (ct_arr <= 500))

    return stats


def _generate_hu_histogram(ct_arr: np.ndarray) -> str:
    """Generate HU histogram as a base64-encoded PNG image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # ── Full-range histogram ──────────────────────────────────────
    body_voxels = ct_arr[ct_arr > -500].ravel()
    ax1.hist(body_voxels, bins=200, range=(-200, 1500), color="#58a6ff",
             alpha=0.8, edgecolor="none")
    ax1.set_facecolor("#161b22")
    ax1.set_xlabel("Hounsfield Units (HU)", color="#e6edf3", fontsize=10)
    ax1.set_ylabel("Voxel Count", color="#e6edf3", fontsize=10)
    ax1.set_title("HU Distribution (Body)", color="#e6edf3", fontsize=12, fontweight="bold")
    ax1.tick_params(colors="#8b949e")

    # Add tissue bands
    bands = [
        (-190, -30,  "#f0e68c30", "Fat"),
        (10, 40,     "#ff634730", "Muscle"),
        (40, 70,     "#da70d630", "Liver"),
        (300, 1500,  "#87ceeb30", "Bone"),
    ]
    for lo, hi, color, label in bands:
        ax1.axvspan(lo, hi, color=color, label=label)
    ax1.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#e6edf3")

    # ── Soft tissue detail ────────────────────────────────────────
    soft = ct_arr[(ct_arr >= -50) & (ct_arr <= 150)].ravel()
    if len(soft) > 0:
        ax2.hist(soft, bins=150, range=(-50, 150), color="#f97583",
                 alpha=0.8, edgecolor="none")
    ax2.set_facecolor("#161b22")
    ax2.set_xlabel("Hounsfield Units (HU)", color="#e6edf3", fontsize=10)
    ax2.set_ylabel("Voxel Count", color="#e6edf3", fontsize=10)
    ax2.set_title("Soft Tissue Detail (-50 to 150 HU)", color="#e6edf3",
                  fontsize=12, fontweight="bold")
    ax2.tick_params(colors="#8b949e")
    # Mark organ ranges
    ax2.axvspan(-10, 15, color="#4ecdc430", label="Water/Fluid")
    ax2.axvspan(40, 70, color="#da70d630", label="Liver")
    ax2.axvspan(20, 45, color="#ffa50030", label="Kidney")
    ax2.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#e6edf3")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _generate_tissue_pie(stats: TissueStats) -> str:
    """Generate tissue composition pie chart as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    sizes = []
    colors_list = []

    tissue_data = [
        ("Lung Air",       stats.lung_volume_ml,       "#58a6ff"),
        ("Fat",            stats.fat_volume_ml,         "#f0e68c"),
        ("Muscle",         stats.muscle_volume_ml,      "#ff6347"),
        ("Soft Tissue",    stats.soft_tissue_volume_ml, "#da70d6"),
        ("Bone",           stats.bone_volume_ml,        "#87ceeb"),
        ("Fluid/Water",    stats.water_fluid_volume_ml, "#4ecdc4"),
        ("Calcifications", stats.calcification_volume_ml, "#ffa500"),
    ]

    for label, vol, color in tissue_data:
        if vol > 10:  # Skip negligible
            labels.append(f"{label}\n{vol:.0f} mL")
            sizes.append(vol)
            colors_list.append(color)

    if not sizes:
        return ""

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_list, autopct="%1.1f%%",
        startangle=90, textprops={"color": "#e6edf3", "fontsize": 9},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("#e6edf3")
    ax.set_title("Body Tissue Composition", color="#e6edf3",
                 fontsize=14, fontweight="bold", pad=20)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ═══════════════════════════════════════════════════════════════════════════
# Clinical finding generators
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_lungs(ct_arr: np.ndarray, stats: TissueStats,
                   findings: list[Finding]) -> None:
    """Analyze lung parenchyma for pathology."""
    vox_ml = stats.voxel_volume_mm3 / 1000.0

    if stats.lung_volume_ml < 500:
        # Might be a non-chest scan or severe pathology
        findings.append(Finding(
            category="incidental", organ="Lungs",
            title="Minimal lung tissue in scan",
            detail="Very little lung parenchyma detected. This may indicate the "
                   "scan primarily covers abdomen/pelvis, OR there is significant "
                   "bilateral lung pathology (collapse, effusion, consolidation).",
            severity="mild",
            measurement=f"Lung volume: {stats.lung_volume_ml:.0f} mL",
        ))
        return

    # Emphysema: low-attenuation areas < -950 HU within lung region
    lung_mask = (ct_arr >= -1024) & (ct_arr <= -500)
    lung_vox = ct_arr[lung_mask]
    if len(lung_vox) > 0:
        emphysema_pct = float(np.count_nonzero(lung_vox < -950)) / len(lung_vox) * 100
        if emphysema_pct > 15:
            findings.append(Finding(
                category="abnormal", organ="Lungs",
                title="Significant emphysematous changes",
                detail=f"Approximately {emphysema_pct:.1f}% of lung voxels show "
                       "low attenuation consistent with emphysema (< -950 HU). "
                       "This suggests chronic obstructive pulmonary disease (COPD). "
                       "Clinical correlation recommended.",
                severity="moderate",
                measurement=f"Emphysema index: {emphysema_pct:.1f}%",
            ))
        elif emphysema_pct > 5:
            findings.append(Finding(
                category="abnormal", organ="Lungs",
                title="Mild emphysematous changes",
                detail=f"Approximately {emphysema_pct:.1f}% of lung voxels show "
                       "low attenuation (< -950 HU) suggestive of mild emphysema.",
                severity="mild",
                measurement=f"Emphysema index: {emphysema_pct:.1f}%",
            ))

    # Ground-glass opacity / consolidation: -700 to -300 HU in lung region
    ggo_mask = (ct_arr >= -700) & (ct_arr <= -300) & lung_mask
    ggo_vol = float(np.count_nonzero(ggo_mask)) * vox_ml
    if ggo_vol > 100:
        ggo_pct = ggo_vol / stats.lung_volume_ml * 100 if stats.lung_volume_ml > 0 else 0
        findings.append(Finding(
            category="abnormal", organ="Lungs",
            title="Ground-glass / mixed attenuation areas",
            detail=f"Approximately {ggo_vol:.0f} mL of lung tissue shows "
                   f"intermediate attenuation (-700 to -300 HU), comprising "
                   f"{ggo_pct:.1f}% of total lung volume. This pattern may "
                   "represent ground-glass opacity, interstitial changes, "
                   "or partial consolidation. Clinical correlation with "
                   "symptoms and follow-up imaging recommended.",
            severity="moderate" if ggo_pct > 20 else "mild",
            measurement=f"GGO volume: {ggo_vol:.0f} mL ({ggo_pct:.1f}%)",
        ))

    # Pleural effusion: fluid density near lung bases
    # Check inferior 20% of lung z-range for fluid (-10 to 20 HU)
    z_size = ct_arr.shape[0]
    inferior = ct_arr[int(z_size * 0.7):, :, :]
    effusion_mask = (inferior >= -10) & (inferior <= 20)
    effusion_vol = float(np.count_nonzero(effusion_mask)) * vox_ml
    if effusion_vol > 200:
        findings.append(Finding(
            category="abnormal", organ="Lungs",
            title="Possible pleural effusion",
            detail=f"Fluid-density tissue ({effusion_vol:.0f} mL) detected in the "
                   "inferior thoracic region, possibly representing pleural effusion. "
                   "Bilateral vs unilateral distribution should be assessed. "
                   "May cause restrictive ventilatory impairment.",
            severity="moderate",
            measurement=f"Inferior fluid: {effusion_vol:.0f} mL",
        ))

    # Normal lungs
    if not any(f.organ == "Lungs" for f in findings if f.category == "abnormal"):
        findings.append(Finding(
            category="normal", organ="Lungs",
            title="Lungs appear within normal limits",
            detail=f"Total aerated lung volume is {stats.lung_volume_ml:.0f} mL. "
                   "No significant emphysematous changes, consolidation, or "
                   "pleural effusion detected on automated analysis.",
            severity="normal",
            measurement=f"Lung volume: {stats.lung_volume_ml:.0f} mL",
        ))


def _analyze_bones(ct_arr: np.ndarray, stats: TissueStats,
                   findings: list[Finding]) -> None:
    """Analyze skeletal structures."""
    vox_ml = stats.voxel_volume_mm3 / 1000.0

    # Bone density assessment
    bone_mask = ct_arr > 300
    bone_vox = ct_arr[bone_mask]

    if len(bone_vox) == 0:
        findings.append(Finding(
            category="incidental", organ="Bones",
            title="No significant bone tissue detected",
            detail="No voxels above 300 HU detected. This may indicate a "
                   "non-standard scan or unusual windowing.",
            severity="mild",
        ))
        return

    mean_bone_hu = float(np.mean(bone_vox))
    dense_pct = float(np.count_nonzero(bone_vox > 700)) / len(bone_vox) * 100

    # Osteoporosis screening: reduced cortical bone density
    if mean_bone_hu < 400:
        findings.append(Finding(
            category="abnormal", organ="Bones",
            title="Reduced overall bone density",
            detail=f"Mean bone attenuation is {mean_bone_hu:.0f} HU (expected > 500 HU "
                   "for normal cortical bone). This may suggest osteopenia or "
                   "osteoporosis. DEXA scan correlation recommended for definitive "
                   "assessment.",
            severity="moderate",
            measurement=f"Mean bone HU: {mean_bone_hu:.0f}",
        ))
    elif mean_bone_hu < 500:
        findings.append(Finding(
            category="risk", organ="Bones",
            title="Borderline bone density",
            detail=f"Mean bone attenuation is {mean_bone_hu:.0f} HU, which is "
                   "borderline. Consider DEXA scan if osteoporosis risk factors "
                   "are present.",
            severity="mild",
            measurement=f"Mean bone HU: {mean_bone_hu:.0f}",
        ))

    # Fracture indicators: look for discontinuities in high-density regions
    # Simple heuristic: check for fragmented high-density components
    # This is a rough indicator — real fracture detection needs AI
    bone_high = (ct_arr > 500).astype(np.uint8)
    from scipy import ndimage
    labeled, n_components = ndimage.label(bone_high)
    if n_components > 50:
        findings.append(Finding(
            category="risk", organ="Bones",
            title="Highly fragmented bone pattern",
            detail=f"Detected {n_components} separate bone fragments/components. "
                   "This may indicate fractures, degenerative changes, or "
                   "post-surgical hardware. Clinical correlation with patient "
                   "history is essential.",
            severity="moderate",
            measurement=f"Bone components: {n_components}",
        ))

    # Calcifications outside bone
    calc_mask = (ct_arr >= 130) & (ct_arr <= 300)
    calc_vol = float(np.count_nonzero(calc_mask)) * vox_ml
    if calc_vol > 50:
        findings.append(Finding(
            category="abnormal", organ="Bones",
            title="Extra-osseous calcifications detected",
            detail=f"Approximately {calc_vol:.0f} mL of tissue in the 130-300 HU "
                   "range detected outside dense bone regions. This may represent "
                   "vascular calcifications (atherosclerosis), soft tissue "
                   "calcifications, gallstones, or kidney stones.",
            severity="mild" if calc_vol < 200 else "moderate",
            measurement=f"Calcification volume: {calc_vol:.0f} mL",
        ))

    # Normal bone finding
    if not any(f.organ == "Bones" and f.category == "abnormal" for f in findings):
        findings.append(Finding(
            category="normal", organ="Bones",
            title="Skeletal structures appear intact",
            detail=f"Total bone volume is {stats.bone_volume_ml:.0f} mL with mean "
                   f"attenuation of {mean_bone_hu:.0f} HU. Dense cortical bone "
                   f"({dense_pct:.1f}% of total bone) appears within normal range. "
                   "No definite fracture or severely reduced density detected on "
                   "automated analysis.",
            severity="normal",
            measurement=f"Bone volume: {stats.bone_volume_ml:.0f} mL, Mean HU: {mean_bone_hu:.0f}",
        ))


def _analyze_liver(ct_arr: np.ndarray, stats: TissueStats,
                   findings: list[Finding]) -> None:
    """Analyze hepatic parenchyma."""
    vox_ml = stats.voxel_volume_mm3 / 1000.0
    z_size, y_size, x_size = ct_arr.shape

    # Liver ROI: right abdomen, z 20-75%
    z0, z1 = int(z_size * 0.20), int(z_size * 0.75)
    x_mid = int(x_size * 0.45)
    roi = ct_arr[z0:z1, :, x_mid:]

    # Liver tissue (40-70 HU) in the ROI
    liver_mask = (roi >= 40) & (roi <= 70)
    liver_vox = roi[liver_mask]

    if len(liver_vox) < 1000:
        findings.append(Finding(
            category="incidental", organ="Liver",
            title="Insufficient liver tissue for analysis",
            detail="Very little liver-range tissue (40-70 HU) detected in the "
                   "expected right-abdomen region. The scan may not adequately "
                   "cover the liver.",
            severity="mild",
        ))
        return

    mean_liver = float(np.mean(liver_vox))
    std_liver = float(np.std(liver_vox))
    liver_vol = float(np.count_nonzero(liver_mask)) * vox_ml

    # Hepatic steatosis (fatty liver): liver HU < splenic HU or < 40 HU
    # Since we don't have spleen reference, use absolute thresholds
    # Normal liver: 50-65 HU, Fatty liver: < 48 HU
    if mean_liver < 42:
        findings.append(Finding(
            category="abnormal", organ="Liver",
            title="Findings suggestive of hepatic steatosis",
            detail=f"Mean liver attenuation is {mean_liver:.0f} HU (normal > 50 HU). "
                   "Low hepatic attenuation suggests fatty infiltration (steatosis). "
                   "Grade: likely moderate-to-severe. Correlate with liver function "
                   "tests and consider ultrasound for confirmation.",
            severity="moderate",
            measurement=f"Mean liver HU: {mean_liver:.0f}",
        ))
    elif mean_liver < 48:
        findings.append(Finding(
            category="abnormal", organ="Liver",
            title="Borderline low liver attenuation",
            detail=f"Mean liver attenuation is {mean_liver:.0f} HU (normal > 50 HU). "
                   "Mildly reduced attenuation may suggest early/mild fatty "
                   "infiltration. Clinical correlation recommended.",
            severity="mild",
            measurement=f"Mean liver HU: {mean_liver:.0f}",
        ))

    # Heterogeneous liver: high standard deviation
    if std_liver > 18:
        findings.append(Finding(
            category="abnormal", organ="Liver",
            title="Heterogeneous liver parenchyma",
            detail=f"Liver tissue shows high attenuation variability (σ = {std_liver:.1f} HU). "
                   "Heterogeneity may indicate focal lesions, cirrhotic changes, "
                   "or diffuse parenchymal disease. Dedicated liver imaging "
                   "(contrast CT or MRI) may be warranted.",
            severity="mild",
            measurement=f"Liver HU std dev: {std_liver:.1f}",
        ))

    # Hepatomegaly: rough liver volume check
    if liver_vol > 2500:
        findings.append(Finding(
            category="abnormal", organ="Liver",
            title="Possible hepatomegaly",
            detail=f"Estimated liver tissue volume is {liver_vol:.0f} mL "
                   "(normal range: 1000-1800 mL). Enlarged liver may indicate "
                   "hepatic congestion, steatosis, infiltrative disease, or "
                   "cirrhosis. Clinical correlation required.",
            severity="mild",
            measurement=f"Liver volume: {liver_vol:.0f} mL",
        ))

    # Normal liver
    if not any(f.organ == "Liver" and f.category == "abnormal" for f in findings):
        findings.append(Finding(
            category="normal", organ="Liver",
            title="Liver appears within normal limits",
            detail=f"Mean liver attenuation is {mean_liver:.0f} ± {std_liver:.1f} HU "
                   f"with an estimated volume of {liver_vol:.0f} mL. No significant "
                   "steatosis, heterogeneity, or hepatomegaly detected.",
            severity="normal",
            measurement=f"Liver: {mean_liver:.0f} HU, {liver_vol:.0f} mL",
        ))


def _analyze_kidneys(ct_arr: np.ndarray, stats: TissueStats,
                     findings: list[Finding]) -> None:
    """Analyze renal parenchyma."""
    vox_ml = stats.voxel_volume_mm3 / 1000.0
    z_size, y_size, x_size = ct_arr.shape

    # Kidney ROI: z-band 30-70%, both sides
    z0, z1 = int(z_size * 0.30), int(z_size * 0.70)
    roi = ct_arr[z0:z1, :, :]

    # Kidney-range tissue
    kidney_mask = (roi >= 20) & (roi <= 45)
    kidney_vol = float(np.count_nonzero(kidney_mask)) * vox_ml

    # Check for kidney stones: high-density inclusions (> 200 HU)
    # in the posterior renal fossa region
    posterior = roi[:, int(y_size * 0.5):, :]
    stone_mask = (posterior >= 200) & (posterior <= 1200)
    stone_vol = float(np.count_nonzero(stone_mask)) * vox_ml

    if stone_vol > 5:
        findings.append(Finding(
            category="abnormal", organ="Kidneys",
            title="Possible renal/ureteral calculi",
            detail=f"High-density foci ({stone_vol:.1f} mL) detected in the "
                   "posterior abdominal region at kidney level (200-1200 HU). "
                   "These may represent renal stones (nephrolithiasis), "
                   "ureteral calculi, or vascular calcifications. "
                   "Correlation with symptoms (flank pain, hematuria) recommended.",
            severity="moderate" if stone_vol > 20 else "mild",
            measurement=f"High-density foci: {stone_vol:.1f} mL",
        ))

    # Asymmetry: check left vs right kidney volumes
    left_roi = roi[:, :, :int(x_size * 0.45)]
    right_roi = roi[:, :, int(x_size * 0.55):]
    left_kid = float(np.count_nonzero((left_roi >= 20) & (left_roi <= 45))) * vox_ml
    right_kid = float(np.count_nonzero((right_roi >= 20) & (right_roi <= 45))) * vox_ml

    if left_kid > 50 and right_kid > 50:
        ratio = max(left_kid, right_kid) / min(left_kid, right_kid)
        if ratio > 2.0:
            smaller = "left" if left_kid < right_kid else "right"
            findings.append(Finding(
                category="abnormal", organ="Kidneys",
                title="Significant renal asymmetry",
                detail=f"Marked size difference between kidneys: left {left_kid:.0f} mL, "
                       f"right {right_kid:.0f} mL (ratio {ratio:.1f}:1). "
                       f"The {smaller} kidney appears significantly smaller. "
                       "This may indicate chronic kidney disease, renal artery "
                       "stenosis, or congenital hypoplasia on the affected side.",
                severity="moderate",
                measurement=f"L/R ratio: {ratio:.1f}:1",
            ))

    # Hydronephrosis hint: fluid in kidney region
    fluid_mask = (roi >= -10) & (roi <= 15)
    kidney_region = roi[:, int(y_size * 0.3):int(y_size * 0.7), :]
    fluid_in_kid = float(np.count_nonzero(
        (kidney_region >= -10) & (kidney_region <= 15)
    )) * vox_ml
    if fluid_in_kid > 150:
        findings.append(Finding(
            category="risk", organ="Kidneys",
            title="Possible hydronephrosis",
            detail=f"Excess fluid-density tissue ({fluid_in_kid:.0f} mL) detected "
                   "in the renal region. This may indicate hydronephrosis "
                   "(distended renal pelvis). Ultrasound or CT urogram "
                   "recommended for confirmation.",
            severity="mild",
            measurement=f"Fluid in renal region: {fluid_in_kid:.0f} mL",
        ))

    # Normal kidneys
    if not any(f.organ == "Kidneys" and f.category in ("abnormal", "risk") for f in findings):
        findings.append(Finding(
            category="normal", organ="Kidneys",
            title="Kidneys appear within normal limits",
            detail=f"Renal tissue volume estimated at {kidney_vol:.0f} mL "
                   f"(left: {left_kid:.0f} mL, right: {right_kid:.0f} mL). "
                   "No significant calculi, asymmetry, or hydronephrosis detected.",
            severity="normal",
            measurement=f"Kidney volume: {kidney_vol:.0f} mL",
        ))


def _analyze_abdomen(ct_arr: np.ndarray, stats: TissueStats,
                     findings: list[Finding]) -> None:
    """Analyze abdominal soft tissues and vasculature."""
    vox_ml = stats.voxel_volume_mm3 / 1000.0
    z_size = ct_arr.shape[0]

    # Ascites: large fluid collection in abdomen
    abdominal = ct_arr[int(z_size * 0.3):int(z_size * 0.8), :, :]
    fluid_mask = (abdominal >= -10) & (abdominal <= 20)
    fluid_vol = float(np.count_nonzero(fluid_mask)) * vox_ml

    if fluid_vol > 1000:
        findings.append(Finding(
            category="abnormal", organ="Abdomen",
            title="Possible ascites",
            detail=f"Large volume of fluid-density tissue ({fluid_vol:.0f} mL) "
                   "detected in the abdominal cavity. This may represent ascites "
                   "(peritoneal fluid collection), which can be caused by liver "
                   "cirrhosis, heart failure, malignancy, or infection.",
            severity="moderate" if fluid_vol > 2000 else "mild",
            measurement=f"Abdominal fluid: {fluid_vol:.0f} mL",
        ))

    # Visceral fat assessment
    if stats.fat_volume_ml > 5000:
        findings.append(Finding(
            category="risk", organ="Abdomen",
            title="Elevated body fat",
            detail=f"Total fat volume is {stats.fat_volume_ml:.0f} mL. "
                   "Elevated visceral and subcutaneous fat increases risk for "
                   "metabolic syndrome, cardiovascular disease, and type 2 diabetes.",
            severity="mild",
            measurement=f"Fat volume: {stats.fat_volume_ml:.0f} mL",
        ))

    # Vascular calcifications
    if stats.calcification_volume_ml > 100:
        findings.append(Finding(
            category="abnormal", organ="Vasculature",
            title="Vascular calcifications detected",
            detail=f"Approximately {stats.calcification_volume_ml:.0f} mL of "
                   "calcified tissue (130-300 HU) detected, likely representing "
                   "aortic and/or iliac artery atherosclerotic calcifications. "
                   "This is a marker of cardiovascular disease risk.",
            severity="moderate" if stats.calcification_volume_ml > 300 else "mild",
            measurement=f"Calcification: {stats.calcification_volume_ml:.0f} mL",
        ))

    # Bowel gas pattern
    bowel_gas_mask = (ct_arr >= -900) & (ct_arr <= -600)
    # Exclude lung region for bowel gas (only count in abdomen)
    abd_gas = ct_arr[int(z_size * 0.5):, :, :]
    gas_vol = float(np.count_nonzero((abd_gas >= -900) & (abd_gas <= -600))) * vox_ml

    if gas_vol > 3000:
        findings.append(Finding(
            category="risk", organ="Abdomen",
            title="Prominent bowel gas pattern",
            detail=f"Elevated intraluminal gas detected ({gas_vol:.0f} mL) in the "
                   "abdominal region. While often physiological, markedly distended "
                   "bowel with excessive gas may suggest ileus, bowel obstruction, "
                   "or functional dysmotility.",
            severity="mild",
            measurement=f"Abdominal gas: {gas_vol:.0f} mL",
        ))


def _analyze_spine(ct_arr: np.ndarray, stats: TissueStats,
                   findings: list[Finding]) -> None:
    """Analyze spinal column for degenerative changes."""
    z_size, y_size, x_size = ct_arr.shape
    vox_ml = stats.voxel_volume_mm3 / 1000.0

    # Spine ROI: center 30% of x, posterior 50% of y
    x0, x1 = int(x_size * 0.35), int(x_size * 0.65)
    y0 = int(y_size * 0.5)
    spine_roi = ct_arr[:, y0:, x0:x1]

    # Vertebral body density (trabecular bone: 300-500 HU region)
    trabecular = spine_roi[(spine_roi >= 100) & (spine_roi <= 500)]
    if len(trabecular) > 1000:
        mean_trab = float(np.mean(trabecular))
        if mean_trab < 150:
            findings.append(Finding(
                category="abnormal", organ="Spine",
                title="Reduced vertebral body density",
                detail=f"Mean trabecular bone density in spinal region is "
                       f"{mean_trab:.0f} HU. Values below 150 HU are associated "
                       "with increased fracture risk and may indicate osteoporosis. "
                       "DEXA scan recommended for definitive assessment.",
                severity="moderate",
                measurement=f"Vertebral density: {mean_trab:.0f} HU",
            ))
        elif mean_trab < 200:
            findings.append(Finding(
                category="risk", organ="Spine",
                title="Slightly reduced vertebral body density",
                detail=f"Mean vertebral trabecular density is {mean_trab:.0f} HU. "
                       "This is in the lower range and may warrant monitoring, "
                       "especially in patients with osteoporosis risk factors.",
                severity="mild",
                measurement=f"Vertebral density: {mean_trab:.0f} HU",
            ))

    # Disc space narrowing indicator: look for bone-to-bone contact
    # (Very rough — true disc assessment needs sagittal reformats + AI)
    # We'll just note that spinal structures are present
    if not any(f.organ == "Spine" for f in findings):
        findings.append(Finding(
            category="normal", organ="Spine",
            title="Spinal column intact on automated analysis",
            detail="No significant reduction in vertebral body density detected. "
                   "Detailed assessment of disc spaces, facet joints, and neural "
                   "foramina requires radiologist review.",
            severity="normal",
        ))


def _determine_scan_coverage(ct_arr: np.ndarray, spacing: tuple) -> str:
    """Determine approximate body region covered by the scan."""
    z_extent_mm = ct_arr.shape[0] * spacing[2]
    has_lung = float(np.count_nonzero((ct_arr >= -950) & (ct_arr <= -650))) > 100000
    has_pelvis = ct_arr.shape[0] > 200  # rough

    if z_extent_mm > 800 and has_lung:
        return "Chest, Abdomen, and Pelvis (whole-body coverage)"
    elif z_extent_mm > 500 and has_lung:
        return "Chest and Abdomen"
    elif z_extent_mm > 400 and not has_lung:
        return "Abdomen and Pelvis"
    elif has_lung:
        return "Chest (partial)"
    else:
        return f"Regional scan ({z_extent_mm:.0f} mm z-extent)"


# ═══════════════════════════════════════════════════════════════════════════
# HTML report generation
# ═══════════════════════════════════════════════════════════════════════════

def _severity_color(severity: str) -> str:
    """Return CSS color for severity level."""
    return {
        "normal":   "#3fb950",
        "mild":     "#d29922",
        "moderate": "#f97583",
        "severe":   "#f85149",
        "critical": "#da3633",
    }.get(severity, "#8b949e")


def _category_icon(category: str) -> str:
    """Return HTML icon for finding category."""
    return {
        "normal":     "&#10004;",   # ✔
        "abnormal":   "&#9888;",    # ⚠
        "risk":       "&#9889;",    # ⚡
        "incidental": "&#8505;",    # ℹ
    }.get(category, "&#8226;")     # •


def _build_html_report(report: MedicalReport, hist_b64: str, pie_b64: str) -> str:
    """Build the full HTML medical findings report."""
    p = report.patient

    # ── Patient info table ────────────────────────────────────
    patient_html = f"""
    <div class="info-grid">
        <div class="info-card">
            <h3>Patient Demographics</h3>
            <table class="info-table">
                <tr><td class="label">Patient ID</td><td>{p.patient_id}</td></tr>
                <tr><td class="label">Name</td><td>{p.patient_name}</td></tr>
                <tr><td class="label">Age</td><td>{p.patient_age}</td></tr>
                <tr><td class="label">Sex</td><td>{p.patient_sex}</td></tr>
                <tr><td class="label">Weight</td><td>{p.patient_weight}</td></tr>
            </table>
        </div>
        <div class="info-card">
            <h3>Study Information</h3>
            <table class="info-table">
                <tr><td class="label">Study Date</td><td>{p.study_date}</td></tr>
                <tr><td class="label">Description</td><td>{p.study_description}</td></tr>
                <tr><td class="label">Body Part</td><td>{p.body_part}</td></tr>
                <tr><td class="label">institution</td><td>{p.institution}</td></tr>
                <tr><td class="label">Referring Physician</td><td>{p.referring_physician}</td></tr>
                <tr><td class="label">Accession #</td><td>{p.accession_number}</td></tr>
            </table>
        </div>
        <div class="info-card">
            <h3>Technical Parameters</h3>
            <table class="info-table">
                <tr><td class="label">Scanner</td><td>{p.manufacturer} {p.model_name}</td></tr>
                <tr><td class="label">Protocol</td><td>{p.protocol}</td></tr>
                <tr><td class="label">kVp</td><td>{p.kvp}</td></tr>
                <tr><td class="label">Tube Current</td><td>{p.tube_current}</td></tr>
                <tr><td class="label">Slice Thickness</td><td>{p.slice_thickness}</td></tr>
                <tr><td class="label">Kernel</td><td>{p.convolution_kernel}</td></tr>
                <tr><td class="label">Contrast</td><td>{p.contrast_agent}</td></tr>
            </table>
        </div>
        <div class="info-card">
            <h3>Volume Information</h3>
            <table class="info-table">
                <tr><td class="label">Volume Size</td><td>{report.volume_size[0]} × {report.volume_size[1]} × {report.volume_size[2]}</td></tr>
                <tr><td class="label">Spacing</td><td>{report.spacing[0]:.3f} × {report.spacing[1]:.3f} × {report.spacing[2]:.3f} mm</td></tr>
                <tr><td class="label">HU Range</td><td>[{report.hu_min:.0f}, {report.hu_max:.0f}]</td></tr>
                <tr><td class="label">HU Mean ± SD</td><td>{report.hu_mean:.1f} ± {report.hu_std:.1f}</td></tr>
                <tr><td class="label">Scan Coverage</td><td>{report.scan_coverage}</td></tr>
                <tr><td class="label">Image Quality</td><td>{report.scan_quality}</td></tr>
            </table>
        </div>
    </div>"""

    # ── Findings grouped by category ──────────────────────────
    categories_order = ["abnormal", "risk", "incidental", "normal"]
    category_labels = {
        "abnormal":   "Abnormal Findings",
        "risk":       "Potential Risks",
        "incidental": "Incidental Observations",
        "normal":     "Normal Findings",
    }

    findings_html = ""
    for cat in categories_order:
        cat_findings = [f for f in report.findings if f.category == cat]
        if not cat_findings:
            continue

        findings_html += f'<h2 class="cat-header cat-{cat}">{_category_icon(cat)} {category_labels[cat]}</h2>\n'
        for f in cat_findings:
            sev_color = _severity_color(f.severity)
            measurement_html = f'<div class="measurement">{f.measurement}</div>' if f.measurement else ""
            findings_html += f"""
            <div class="finding finding-{f.category}">
                <div class="finding-header">
                    <span class="organ-tag">{f.organ}</span>
                    <span class="finding-title">{f.title}</span>
                    <span class="severity" style="color: {sev_color}; border-color: {sev_color}">
                        {f.severity.upper()}
                    </span>
                </div>
                <div class="finding-detail">{f.detail}</div>
                {measurement_html}
            </div>"""

    # ── Tissue table ──────────────────────────────────────────
    t = report.tissue
    body_vol = t.total_body_voxels * t.voxel_volume_mm3 / 1000.0
    tissue_rows = [
        ("Lung Parenchyma", f"{t.lung_volume_ml:.0f}"),
        ("Fat (subcutaneous + visceral)", f"{t.fat_volume_ml:.0f}"),
        ("Muscle", f"{t.muscle_volume_ml:.0f}"),
        ("Soft Tissue (20-80 HU)", f"{t.soft_tissue_volume_ml:.0f}"),
        ("Water / Fluid", f"{t.water_fluid_volume_ml:.0f}"),
        ("Bone (> 300 HU)", f"{t.bone_volume_ml:.0f}"),
        ("Dense Bone (> 700 HU)", f"{t.dense_bone_volume_ml:.0f}"),
        ("Calcifications (130-300 HU)", f"{t.calcification_volume_ml:.0f}"),
        ("Total Body", f"{body_vol:.0f}"),
    ]
    tissue_table_rows = "\n".join(
        f"<tr><td>{name}</td><td>{vol} mL</td></tr>"
        for name, vol in tissue_rows
    )

    # ── Summary statistics ────────────────────────────────────
    n_abnormal = sum(1 for f in report.findings if f.category == "abnormal")
    n_risk = sum(1 for f in report.findings if f.category == "risk")
    n_normal = sum(1 for f in report.findings if f.category == "normal")
    n_total = len(report.findings)

    summary_class = "summary-normal"
    if n_abnormal > 3:
        summary_class = "summary-severe"
        summary_text = f"Multiple abnormalities detected ({n_abnormal} findings). Clinical review strongly recommended."
    elif n_abnormal > 0:
        summary_class = "summary-warning"
        summary_text = f"{n_abnormal} abnormal finding(s) detected. Clinical correlation recommended."
    else:
        summary_text = "No significant abnormalities detected on automated analysis."

    # ── Assemble full HTML ────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medical Findings Report — {p.patient_id}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', -apple-system, Tahoma, sans-serif;
        background: #0d1117;
        color: #e6edf3;
        max-width: 1300px;
        margin: 0 auto;
        padding: 20px 30px;
        line-height: 1.6;
    }}
    h1 {{
        color: #58a6ff;
        font-size: 28px;
        border-bottom: 2px solid #30363d;
        padding-bottom: 12px;
        margin-bottom: 8px;
    }}
    h2 {{
        color: #79c0ff;
        margin: 30px 0 15px 0;
        font-size: 20px;
    }}
    h3 {{
        color: #58a6ff;
        font-size: 15px;
        margin-bottom: 10px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
    }}
    .disclaimer {{
        background: #1c1917;
        border: 1px solid #f97583;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 15px 0 25px 0;
        font-size: 13px;
        color: #f97583;
        line-height: 1.5;
    }}
    .disclaimer strong {{ color: #f85149; }}
    .summary-box {{
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 16px;
        font-weight: 600;
    }}
    .summary-normal {{ background: #0d2818; border: 1px solid #238636; color: #3fb950; }}
    .summary-warning {{ background: #2d1b00; border: 1px solid #9e6a03; color: #d29922; }}
    .summary-severe {{ background: #2d0000; border: 1px solid #da3633; color: #f85149; }}
    .info-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin: 20px 0;
    }}
    .info-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
    }}
    .info-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .info-table td {{
        padding: 5px 8px;
        font-size: 13px;
        border-bottom: 1px solid #21262d;
    }}
    .info-table .label {{
        color: #8b949e;
        width: 45%;
        font-weight: 500;
    }}
    .cat-header {{
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 25px;
    }}
    .cat-abnormal {{ background: #2d000030; border-left: 4px solid #f85149; }}
    .cat-risk {{ background: #2d1b0030; border-left: 4px solid #d29922; }}
    .cat-incidental {{ background: #0d281830; border-left: 4px solid #58a6ff; }}
    .cat-normal {{ background: #0d281830; border-left: 4px solid #3fb950; }}
    .finding {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px 18px;
        margin: 10px 0;
    }}
    .finding-abnormal {{ border-left: 3px solid #f85149; }}
    .finding-risk {{ border-left: 3px solid #d29922; }}
    .finding-incidental {{ border-left: 3px solid #58a6ff; }}
    .finding-normal {{ border-left: 3px solid #3fb950; }}
    .finding-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }}
    .organ-tag {{
        background: #21262d;
        color: #79c0ff;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    .finding-title {{
        font-weight: 600;
        font-size: 15px;
        flex: 1;
    }}
    .severity {{
        font-size: 11px;
        font-weight: 700;
        padding: 2px 10px;
        border: 1px solid;
        border-radius: 12px;
        letter-spacing: 0.5px;
    }}
    .finding-detail {{
        color: #c9d1d9;
        font-size: 14px;
        line-height: 1.7;
    }}
    .measurement {{
        margin-top: 8px;
        font-family: 'Cascadia Code', 'Consolas', monospace;
        font-size: 13px;
        color: #58a6ff;
        background: #0d1117;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
    }}
    .tissue-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }}
    .tissue-table th, .tissue-table td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #30363d;
        font-size: 14px;
    }}
    .tissue-table th {{
        background: #161b22;
        color: #58a6ff;
        font-weight: 600;
    }}
    .tissue-table tr:hover {{
        background: #161b22;
    }}
    .tissue-table tr:last-child {{
        font-weight: 700;
        border-top: 2px solid #30363d;
    }}
    .charts {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }}
    .charts img {{
        width: 100%;
        border-radius: 8px;
        border: 1px solid #30363d;
    }}
    .chart-full img {{
        width: 100%;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin: 10px 0;
    }}
    .footer {{
        text-align: center;
        color: #484f58;
        font-size: 12px;
        margin-top: 40px;
        padding: 15px 0;
        border-top: 1px solid #21262d;
    }}
    @media (max-width: 768px) {{
        .info-grid, .charts {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

<h1>&#128200; Medical Findings Report</h1>
<p style="color: #8b949e; margin-bottom: 5px;">
    Patient: <strong style="color:#e6edf3">{p.patient_id}</strong>
    &nbsp;|&nbsp; Date: {p.study_date}
    &nbsp;|&nbsp; Analysis time: {report.analysis_time:.1f} s
</p>

<div class="disclaimer">
    <strong>&#9888; DISCLAIMER:</strong> This report is generated by automated computational
    analysis of CT imaging data. It is <strong>NOT</strong> a substitute for professional
    radiological interpretation. All findings, measurements, and assessments require
    validation by a qualified radiologist or physician. Do not make clinical decisions
    based solely on this automated report.
</div>

<div class="summary-box {summary_class}">
    {summary_text}
    &nbsp;&mdash;&nbsp; {n_total} findings: {n_abnormal} abnormal, {n_risk} risk, {n_normal} normal
</div>

<h2>&#128100; Patient & Study Information</h2>
{patient_html}

<h2>&#128269; Key Findings</h2>
{findings_html}

<h2>&#129516; Tissue Composition</h2>
<table class="tissue-table">
    <tr><th>Tissue Type</th><th>Volume</th></tr>
    {tissue_table_rows}
</table>

<div class="chart-full">
    <h2>&#128202; HU Distribution</h2>
    <img src="{hist_b64}" alt="HU Histogram">
</div>

{"<div class=charts><img src=" + chr(34) + pie_b64 + chr(34) + " alt=Tissue Composition></div>" if pie_b64 else ""}

<div class="footer">
    Generated by MedRecon Engine &mdash; Medical Findings Report<br>
    Automated CT analysis &mdash; For research purposes only
</div>

</body>
</html>"""

    return html


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_medical_report(
    dicom_dir: str | Path,
    output_dir: str | Path,
    *,
    ct_image: Optional[sitk.Image] = None,
) -> Path:
    """Generate a comprehensive medical findings report from DICOM data.

    Parameters
    ----------
    dicom_dir : str | Path
        Path to the DICOM directory.
    output_dir : str | Path
        Output directory for the report.
    ct_image : sitk.Image | None
        Pre-loaded CT volume (avoids reloading).
        If None, will load from dicom_dir.

    Returns
    -------
    Path to the generated HTML report.
    """
    t0 = time.perf_counter()
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Generating medical findings report …")

    # ── Step 1: Extract DICOM metadata ────────────────────────────
    _log.info("  [1/5] Reading DICOM headers …")
    patient_info = _extract_patient_info(dicom_dir)

    # ── Step 2: Load CT volume ────────────────────────────────────
    if ct_image is None:
        _log.info("  [2/5] Loading CT volume …")
        from medrecon_engine.core.volume_loader import VolumeLoader
        loader = VolumeLoader()
        ct_image = loader.load(directory=dicom_dir)
    else:
        _log.info("  [2/5] Using pre-loaded CT volume")

    ct_arr = sitk.GetArrayViewFromImage(ct_image)
    spacing = ct_image.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    # ── Step 3: Compute tissue statistics ─────────────────────────
    _log.info("  [3/5] Analyzing tissue composition …")
    tissue_stats = _compute_tissue_stats(ct_arr, voxel_vol_mm3)

    # ── Step 4: Generate findings ─────────────────────────────────
    _log.info("  [4/5] Generating clinical findings …")
    findings: list[Finding] = []

    _analyze_lungs(ct_arr, tissue_stats, findings)
    _analyze_bones(ct_arr, tissue_stats, findings)
    _analyze_liver(ct_arr, tissue_stats, findings)
    _analyze_kidneys(ct_arr, tissue_stats, findings)
    _analyze_abdomen(ct_arr, tissue_stats, findings)
    _analyze_spine(ct_arr, tissue_stats, findings)

    # Assess scan quality
    scan_quality = "Adequate"
    if ct_arr.shape[2] < 256:
        scan_quality = "Limited FOV"
    noise_roi = ct_arr[ct_arr.shape[0] // 2, :, :]
    noise_std = float(np.std(noise_roi[(noise_roi > 20) & (noise_roi < 60)]))
    if noise_std > 25:
        scan_quality = "Noisy (consider reduced dose or thick slices)"

    # Build report data
    report = MedicalReport(
        patient=patient_info,
        tissue=tissue_stats,
        findings=findings,
        scan_quality=scan_quality,
        scan_coverage=_determine_scan_coverage(ct_arr, spacing),
        volume_size=ct_image.GetSize(),
        spacing=spacing,
        hu_min=float(np.min(ct_arr)),
        hu_max=float(np.max(ct_arr)),
        hu_mean=float(np.mean(ct_arr)),
        hu_std=float(np.std(ct_arr)),
        analysis_time=time.perf_counter() - t0,
    )

    # ── Step 5: Generate visuals & HTML ───────────────────────────
    _log.info("  [5/5] Building HTML report …")
    hist_b64 = _generate_hu_histogram(ct_arr)
    pie_b64 = _generate_tissue_pie(tissue_stats)

    html = _build_html_report(report, hist_b64, pie_b64)
    report_path = output_dir / "medical_findings.html"
    report_path.write_text(html, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    _log.info("Medical findings report generated in %.1f s: %s", elapsed, report_path)

    # Also print summary to console
    n_abnormal = sum(1 for f in findings if f.category == "abnormal")
    n_risk = sum(1 for f in findings if f.category == "risk")
    _log.info("  Findings: %d total — %d abnormal, %d risk, %d normal",
              len(findings), n_abnormal, n_risk,
              sum(1 for f in findings if f.category == "normal"))

    return report_path
