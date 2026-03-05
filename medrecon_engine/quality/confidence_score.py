"""
medrecon_engine.quality.confidence_score
==========================================
Compute a composite confidence score for a reconstructed mesh.

The score aggregates:
- **Topology** (manifold, watertight)          → 30 %
- **Geometry** (aspect ratio, edge uniformity) → 30 %
- **HU model quality** (histogram peak SNR)    → 20 %
- **Resolution** (original voxel vs target)    → 20 %

Output: a float in [0.0, 1.0] where ≥0.80 = surgical-ready,
0.60–0.80 = usable with caveats, <0.60 = reject / manual review.

Deterministic: same inputs → same score, always.
"""

from __future__ import annotations

from dataclasses import dataclass

from medrecon_engine.mesh.mesh_validator import MeshValidationReport
from medrecon_engine.quality.surface_metrics import SurfaceMetrics
from medrecon_engine.hu_model.hu_estimator import HUEstimationResult
from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.audit.logger import get_logger

log = get_logger(__name__)


@dataclass
class ConfidenceReport:
    """Breakdown of the confidence score."""

    total: float = 0.0
    topology_score: float = 0.0
    geometry_score: float = 0.0
    hu_score: float = 0.0
    resolution_score: float = 0.0

    grade: str = "REJECT"   # SURGICAL | USABLE | REJECT

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "topology": round(self.topology_score, 4),
            "geometry": round(self.geometry_score, 4),
            "hu_model": round(self.hu_score, 4),
            "resolution": round(self.resolution_score, 4),
            "grade": self.grade,
        }


class ConfidenceScorer:
    """Compute composite confidence for a pipeline run."""

    # Weights
    W_TOPO = 0.30
    W_GEOM = 0.30
    W_HU = 0.20
    W_RES = 0.20

    def __init__(self, config: PrecisionConfig | None = None):
        self.cfg = config or PrecisionConfig()

    def compute(
        self,
        validation: MeshValidationReport,
        metrics: SurfaceMetrics,
        hu_estimation: HUEstimationResult | None = None,
        original_spacing: tuple[float, float, float] | None = None,
    ) -> ConfidenceReport:
        """Compute the confidence score.

        Parameters
        ----------
        validation : MeshValidationReport
        metrics : SurfaceMetrics
        hu_estimation : HUEstimationResult, optional
        original_spacing : (x, y, z) of the raw DICOM, optional

        Returns
        -------
        ConfidenceReport
        """
        r = ConfidenceReport()

        # ── Topology (0–1) ───────────────────────────────────────────── #
        topo = 1.0
        if not validation.is_manifold:
            topo -= 0.4
        if validation.num_degenerate_faces > 0:
            topo -= min(0.3, validation.num_degenerate_faces / max(validation.num_faces, 1))
        if validation.errors:
            topo -= 0.3
        r.topology_score = max(0.0, topo)

        # ── Geometry (0–1) ───────────────────────────────────────────── #
        geom = 1.0
        if metrics.aspect_ratio_max > self.cfg.max_aspect_ratio:
            geom -= 0.3
        if metrics.aspect_ratio_mean > 5.0:
            geom -= 0.2
        if metrics.edge_length_std > metrics.edge_length_mean * 2.0:
            geom -= 0.2  # very non-uniform edge lengths
        geom *= metrics.normal_consistency  # scale by normal quality
        r.geometry_score = max(0.0, geom)

        # ── HU model quality (0–1) ───────────────────────────────────── #
        if hu_estimation is not None:
            # How close is detected peak to expected?
            peak_err = abs(hu_estimation.detected_peak - hu_estimation.profile.expected_peak)
            # Normalise: 0 error → 1.0, 500 HU error → 0.0
            r.hu_score = max(0.0, 1.0 - peak_err / 500.0)
        else:
            r.hu_score = 0.5  # no estimation data → neutral

        # ── Resolution (0–1) ──────────────────────────────────────────── #
        if original_spacing is not None:
            max_sp = max(original_spacing)
            target = self.cfg.strict_precision_mm
            if max_sp <= target:
                r.resolution_score = 1.0
            else:
                r.resolution_score = max(0.0, 1.0 - (max_sp - target) / target)
        else:
            r.resolution_score = 0.7  # unknown → conservative

        # ── Composite ──────────────────────────────────────────────────── #
        r.total = (
            self.W_TOPO * r.topology_score
            + self.W_GEOM * r.geometry_score
            + self.W_HU * r.hu_score
            + self.W_RES * r.resolution_score
        )

        # Grade
        if r.total >= 0.80:
            r.grade = "SURGICAL"
        elif r.total >= self.cfg.min_confidence_score:
            r.grade = "USABLE"
        else:
            r.grade = "REJECT"

        log.info(
            "Confidence: %.2f (%s)  [topo=%.2f geom=%.2f hu=%.2f res=%.2f]",
            r.total, r.grade,
            r.topology_score, r.geometry_score, r.hu_score, r.resolution_score,
        )
        return r
