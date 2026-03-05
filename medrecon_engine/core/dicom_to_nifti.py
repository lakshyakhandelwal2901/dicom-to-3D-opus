"""
medrecon_engine.core.dicom_to_nifti
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convert a DICOM series folder into a single NIfTI volume (.nii.gz).

TotalSegmentator expects a NIfTI file as input, so this module bridges
the gap between raw DICOM directories and the AI segmenter.
"""

from __future__ import annotations

import SimpleITK as sitk
from pathlib import Path

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)


def dicom_to_nifti(dicom_folder: str | Path, output_path: str | Path) -> Path:
    """Read a DICOM series and write it as NIfTI (.nii.gz).

    Parameters
    ----------
    dicom_folder : str | Path
        Directory containing DICOM files (may contain sub-folders;
        the first series found by GDCM is used).
    output_path : str | Path
        Destination path for the NIfTI volume (e.g. ``scan.nii.gz``).

    Returns
    -------
    Path
        Absolute path to the written NIfTI file.
    """
    dicom_folder = str(dicom_folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _log.info("Converting DICOM → NIfTI  src=%s  dst=%s", dicom_folder, output_path)

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)

    if not dicom_files:
        raise FileNotFoundError(
            f"No DICOM series found in {dicom_folder}"
        )

    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    _log.info(
        "  Volume size=%s  spacing=%.2f×%.2f×%.2f mm",
        image.GetSize(),
        *image.GetSpacing(),
    )

    sitk.WriteImage(image, str(output_path))
    _log.info("  NIfTI written → %s", output_path)

    return output_path.resolve()
