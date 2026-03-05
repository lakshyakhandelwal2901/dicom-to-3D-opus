"""
medrecon_engine.anatomy.ai_segmenter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AI-powered anatomical segmentation using **TotalSegmentator**.

TotalSegmentator is a deep-learning model that segments 117 anatomical
structures from CT scans.  This module wraps the Python API and returns
a directory of per-organ NIfTI label masks.

First run downloads model weights (~1–2 GB). Subsequent runs use the
local cache.

Expected output structure::

    <output_dir>/labels/
        liver.nii.gz
        kidney_left.nii.gz
        pelvis.nii.gz
        femur_left.nii.gz
        vertebrae_L1.nii.gz
        …  (up to 117 structures)
"""

from __future__ import annotations

import os
from pathlib import Path

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)


class AISegmenter:
    """Run TotalSegmentator on a NIfTI CT volume.

    Parameters
    ----------
    output_dir : str | Path
        Root output directory.  Label masks are written into a
        ``labels/`` sub-folder.
    fast : bool
        If *True*, use the faster (lower-resolution) model variant.
        Default *False* for maximum accuracy.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        fast: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fast = fast

    def segment(self, nifti_path: str | Path) -> Path:
        """Run segmentation and return the label directory.

        Parameters
        ----------
        nifti_path : str | Path
            Path to the input CT volume (``*.nii.gz``).

        Returns
        -------
        Path
            Path to the ``labels/`` directory containing per-organ masks.
        """
        from totalsegmentator.python_api import totalsegmentator

        nifti_path = Path(nifti_path)
        label_dir = self.output_dir / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        _log.info("Running TotalSegmentator  input=%s  fast=%s", nifti_path, self.fast)
        _log.info("  Label output → %s", label_dir)

        totalsegmentator(
            input=nifti_path,
            output=label_dir,
            fast=self.fast,
        )

        # Report discovered labels
        labels = sorted(f.stem.replace(".nii", "") for f in label_dir.glob("*.nii.gz"))
        _log.info("  TotalSegmentator produced %d label masks", len(labels))
        if labels:
            _log.info("  Labels: %s", ", ".join(labels[:10]) + (" …" if len(labels) > 10 else ""))

        return label_dir
