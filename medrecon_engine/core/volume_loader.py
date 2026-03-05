"""
medrecon_engine.core.volume_loader
====================================
Load a validated DICOM series into a SimpleITK Image with correct
geometry (spacing, origin, direction cosines).

Strategy
--------
* Use ``sitk.ImageSeriesReader`` for robust, manufacturer-agnostic loading.
* Fall back to manual stacking (pydicom) if SimpleITK can't read the series
  (e.g. non-standard DICOM, broken DICOMDIR).
* Return a ``sitk.Image`` in its original (un-resampled) state — resampling
  is a separate, auditable step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class VolumeLoadError(Exception):
    """Raised when a DICOM series cannot be assembled into a volume."""


class VolumeLoader:
    """Load a DICOM series into a SimpleITK 3-D image."""

    # ------------------------------------------------------------------ #
    #  Public
    # ------------------------------------------------------------------ #
    def load(self, file_paths: list[Path] | None = None,
             directory: str | Path | None = None) -> sitk.Image:
        """Load DICOM either from explicit *file_paths* or by scanning
        *directory*.  Exactly one must be provided.

        Returns
        -------
        sitk.Image
            3-D volume with correct spacing / origin / direction.
        """
        if directory is not None:
            return self._load_from_directory(Path(directory))

        if file_paths is not None and len(file_paths) > 0:
            return self._load_from_paths(file_paths)

        raise VolumeLoadError("Provide either file_paths or directory.")

    # ------------------------------------------------------------------ #
    #  SimpleITK path (preferred)
    # ------------------------------------------------------------------ #
    def _load_from_directory(self, directory: Path) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(directory))

        if not series_ids:
            raise VolumeLoadError(f"No DICOM series found in {directory}")

        # Pick the series with the most files
        best_uid = max(
            series_ids,
            key=lambda uid: len(reader.GetGDCMSeriesFileNames(str(directory), uid)),
        )
        file_names = reader.GetGDCMSeriesFileNames(str(directory), best_uid)
        log.info("VolumeLoader: selected series %s (%d files)", best_uid[:16], len(file_names))

        reader.SetFileNames(file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()
        self._log_image_info(image)
        return image

    def _load_from_paths(self, file_paths: list[Path]) -> sitk.Image:
        """Load from an explicit list of file paths."""
        str_paths = [str(p) for p in file_paths]

        # Sort by ImagePositionPatient[2] (z) for correct ordering
        sorted_paths = self._sort_by_z(str_paths)

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_paths)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        try:
            image = reader.Execute()
        except RuntimeError:
            log.warning("SimpleITK series reader failed — falling back to manual stack")
            image = self._manual_stack(sorted_paths)

        self._log_image_info(image)
        return image

    # ------------------------------------------------------------------ #
    #  Manual fallback
    # ------------------------------------------------------------------ #
    def _manual_stack(self, sorted_paths: list[str]) -> sitk.Image:
        """Stack raw pixel arrays when SimpleITK refuses the series."""
        datasets = [pydicom.dcmread(p) for p in sorted_paths]

        arrays = [ds.pixel_array.astype(np.float64) for ds in datasets]
        volume = np.stack(arrays, axis=0)  # (D, H, W)

        image = sitk.GetImageFromArray(volume)

        ds0 = datasets[0]
        ps = [float(x) for x in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
        st = float(getattr(ds0, "SliceThickness", 1.0))
        image.SetSpacing((ps[1], ps[0], st))  # (x, y, z)

        ipp = [float(x) for x in getattr(ds0, "ImagePositionPatient", [0, 0, 0])]
        image.SetOrigin(tuple(ipp))

        iop = [float(x) for x in getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
        row_cos = np.array(iop[:3])
        col_cos = np.array(iop[3:6])
        slice_cos = np.cross(row_cos, col_cos)
        direction = tuple(float(v) for v in np.concatenate([row_cos, col_cos, slice_cos]))
        image.SetDirection(direction)

        return image

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sort_by_z(paths: list[str]) -> list[str]:
        """Sort DICOM files by z-coordinate of ImagePositionPatient."""
        tagged: list[tuple[float, str]] = []
        for p in paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                z = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0.0
            except Exception:
                z = 0.0
            tagged.append((z, p))
        tagged.sort(key=lambda t: t[0])
        return [p for _, p in tagged]

    @staticmethod
    def _log_image_info(image: sitk.Image) -> None:
        log.info(
            "VolumeLoader: size=%s  spacing=(%.3f, %.3f, %.3f)  origin=(%.1f, %.1f, %.1f)",
            image.GetSize(),
            *image.GetSpacing(),
            *image.GetOrigin(),
        )
