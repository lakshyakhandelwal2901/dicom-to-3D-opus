"""
medrecon_engine.hu_model.hu_estimator
=======================================
Adaptive, per-patient HU threshold estimation via histogram modelling.

Instead of static thresholds (e.g. "bone = 250 HU"), this module:

1.  Computes a 500-bin histogram of the HU volume.
2.  Looks for the dominant peak nearest the profile's ``expected_peak``.
3.  Computes an adaptive lower threshold =  peak − margin  (clamped to
    the profile's hard ``min``).
4.  Upper threshold stays at the profile ``max`` (conservative).

This makes the segmentation **automatically adapt** to inter-patient
variation, different scanner calibrations, and slight contrast differences.

Deterministic: identical input → identical output, always.
"""

from __future__ import annotations

import numpy as np

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.hu_model.hu_profiles import HUProfile, get_profile
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


class HUEstimationResult:
    """Result of the adaptive HU estimation for one anatomy."""

    __slots__ = ("anatomy", "adaptive_min", "adaptive_max",
                 "detected_peak", "profile")

    def __init__(
        self,
        anatomy: str,
        adaptive_min: float,
        adaptive_max: float,
        detected_peak: float,
        profile: HUProfile,
    ):
        self.anatomy = anatomy
        self.adaptive_min = adaptive_min
        self.adaptive_max = adaptive_max
        self.detected_peak = detected_peak
        self.profile = profile

    def __repr__(self) -> str:
        return (
            f"HUEstimationResult(anatomy='{self.anatomy}', "
            f"range=[{self.adaptive_min:.0f}, {self.adaptive_max:.0f}], "
            f"peak={self.detected_peak:.0f})"
        )


class HUEstimator:
    """Adaptive HU range predictor per anatomy per patient."""

    N_BINS: int = 500

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    def estimate(
        self,
        volume_array: np.ndarray,
        anatomy: str,
    ) -> HUEstimationResult:
        """Estimate the optimal HU threshold range for *anatomy*.

        Parameters
        ----------
        volume_array : ndarray
            3-D HU array (float).
        anatomy : str
            Key into ``HU_PROFILES`` (e.g. "bone", "lung").

        Returns
        -------
        HUEstimationResult
        """
        profile = get_profile(anatomy)

        # ── 1) Histogram ────────────────────────────────────────────── #
        # Restrict histogram computation to the profile's hard range
        # to avoid the peak being swamped by air / other tissue.
        in_range = volume_array[
            (volume_array >= profile.min) & (volume_array <= profile.max)
        ]

        if in_range.size == 0:
            log.warning(
                "HU estimator: no voxels in [%.0f, %.0f] for '%s' — using hard limits",
                profile.min, profile.max, anatomy,
            )
            return HUEstimationResult(
                anatomy=anatomy,
                adaptive_min=profile.min,
                adaptive_max=profile.max,
                detected_peak=profile.expected_peak,
                profile=profile,
            )

        hist, bin_edges = np.histogram(in_range, bins=self.N_BINS)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # ── 2) Find dominant peak near expected_peak ─────────────────── #
        #   Weight bins by their count AND proximity to the expected peak
        #   to avoid latching onto an unrelated spike (e.g. table/air).
        proximity = np.exp(
            -0.5 * ((bin_centres - profile.expected_peak) / 200.0) ** 2
        )
        weighted = hist.astype(np.float64) * proximity
        peak_idx = int(np.argmax(weighted))
        detected_peak = float(bin_centres[peak_idx])

        # ── 3) Adaptive thresholds ──────────────────────────────────── #
        margin = self.cfg.hu_adaptive_margin
        adaptive_min = max(profile.min, detected_peak - margin)
        adaptive_max = profile.max  # conservative: keep hard upper limit

        result = HUEstimationResult(
            anatomy=anatomy,
            adaptive_min=adaptive_min,
            adaptive_max=adaptive_max,
            detected_peak=detected_peak,
            profile=profile,
        )

        log.info(
            "HU estimator [%s]: expected_peak=%.0f  detected_peak=%.0f  "
            "adaptive=[%.0f, %.0f]  (voxels in range: %s)",
            anatomy,
            profile.expected_peak,
            detected_peak,
            adaptive_min,
            adaptive_max,
            f"{in_range.size:,}",
        )
        return result
