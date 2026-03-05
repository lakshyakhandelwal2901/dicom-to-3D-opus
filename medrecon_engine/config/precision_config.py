"""
medrecon_engine.config.precision_config
=========================================
Strict, deterministic precision rules for the entire pipeline.

Every numerical threshold in this file has a clinical justification:
- STRICT_PRECISION_MM: minimum spatial resolution for surgical STL models.
- MAX_ALLOWED_SLICE: thicker slices lose z-axis detail on small fractures.
- MAX_ALLOWED_INPLANE: >1 mm in-plane means sub-mm bone cortex is invisible.
- RESAMPLING_TARGET: isotropic 1 mm³ is the sweet-spot for bone/organ CT.

These are HARD limits — the pipeline will REJECT datasets that violate them,
not silently degrade.  Override only with an explicit config dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Spatial precision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

STRICT_PRECISION_MM: float = 1.0
"""Target voxel size after resampling (mm).  The pipeline guarantees that
the final mesh is reconstructed from a volume no coarser than this."""

MAX_ALLOWED_SLICE: float = 1.5
"""Maximum acceptable original slice thickness (mm).
If slices are thicker, the dataset is REJECTED—interpolating thicker
slices down to 1 mm introduces ghost geometry."""

MAX_ALLOWED_INPLANE: float = 1.0
"""Maximum acceptable original in-plane pixel spacing (mm)."""

RESAMPLING_TARGET: tuple[float, float, float] = (1.0, 1.0, 1.0)
"""Target isotropic spacing (x, y, z) in mm for the resampled volume."""

RESAMPLING_INTERPOLATION: Literal["linear", "bspline", "nearest"] = "linear"
"""Interpolation method for resampling.  'linear' is deterministic and safe
for HU-valued data; 'bspline' is smoother but may introduce HU ringing."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  HU calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

HU_CLIP_MIN: float = -1024.0
HU_CLIP_MAX: float = 3071.0

HU_ADAPTIVE_MARGIN: float = 300.0
"""When the HU estimator finds a histogram peak, the adaptive lower
threshold is set to  peak − this margin (clamped to profile min).
Widened from 200 → 300 to preserve thinner cortical bone."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Segmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

MAX_RETAINED_COMPONENTS: int = 5
"""After connected-component labelling, keep at most this many of the
largest components.  This removes dust / noise fragments while keeping
multi-part anatomy (e.g. bilateral femurs)."""

MIN_COMPONENT_VOLUME_MM3: float = 500.0
"""Absolute minimum volume (mm³) for a component to survive filtering.
Applied AFTER the top-N filter — even among the top 5, tiny fragments
are discarded."""

MORPHOLOGICAL_CLOSING_RADIUS: int = 2
"""Ball structuring element radius for binary closing (voxels)."""

GAUSSIAN_SIGMA_MM: float = 0.5
"""Pre-smoothing sigma in mm (applied anisotropically to match spacing)."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Mesh generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

MARCHING_CUBES_ISO: float = 0.5
"""Iso-surface value for marching cubes on the binary mask."""

MESH_SMOOTH_ITERATIONS: int = 25
"""Windowed-sinc (Taubin) smoothing iterations.  Conservative.
Higher = smoother, but risk losing small fracture lines."""

MESH_SMOOTH_PASSBAND: float = 0.01
"""Passband for Taubin smoothing filter.  Small values = more smoothing."""

MESH_DECIMATE_TARGET_RATIO: float = 0.50
"""Keep 50 % of triangles after quadric decimation."""

MESH_FILL_HOLES_SIZE: float = 100.0
"""Fill boundary edges up to this size (# edges) during repair."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Quality thresholds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

MIN_CONFIDENCE_SCORE: float = 0.60
"""Pipeline will WARN (not halt) if the composite confidence drops below this."""

MAX_ASPECT_RATIO: float = 30.0
"""Triangle aspect-ratio ceiling.  Triangles worse than this degrade FEM analysis."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

STL_BINARY: bool = True
"""Write binary STL (smaller, faster) rather than ASCII."""

COORDINATE_SYSTEM: Literal["LPS", "RAS"] = "LPS"
"""LPS = DICOM native.  RAS = common in neuroimaging."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Supported anatomies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

SUPPORTED_ANATOMIES: tuple[str, ...] = (
    "bone",
    "lung",
    "soft_tissue",
    "brain",
    "vascular",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Frozen config dataclass (for passing around)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable snapshot of all precision parameters.

    Use ``PrecisionConfig()`` for defaults, or pass overrides::

        cfg = PrecisionConfig(strict_precision_mm=0.5)
    """

    strict_precision_mm: float = STRICT_PRECISION_MM
    max_allowed_slice: float = MAX_ALLOWED_SLICE
    max_allowed_inplane: float = MAX_ALLOWED_INPLANE
    resampling_target: tuple[float, float, float] = RESAMPLING_TARGET
    resampling_interpolation: str = RESAMPLING_INTERPOLATION

    hu_clip_min: float = HU_CLIP_MIN
    hu_clip_max: float = HU_CLIP_MAX
    hu_adaptive_margin: float = HU_ADAPTIVE_MARGIN

    max_retained_components: int = MAX_RETAINED_COMPONENTS
    min_component_volume_mm3: float = MIN_COMPONENT_VOLUME_MM3
    morphological_closing_radius: int = MORPHOLOGICAL_CLOSING_RADIUS
    gaussian_sigma_mm: float = GAUSSIAN_SIGMA_MM

    marching_cubes_iso: float = MARCHING_CUBES_ISO
    mesh_smooth_iterations: int = MESH_SMOOTH_ITERATIONS
    mesh_smooth_passband: float = MESH_SMOOTH_PASSBAND
    mesh_decimate_target_ratio: float = MESH_DECIMATE_TARGET_RATIO
    mesh_fill_holes_size: float = MESH_FILL_HOLES_SIZE

    min_confidence_score: float = MIN_CONFIDENCE_SCORE
    max_aspect_ratio: float = MAX_ASPECT_RATIO

    stl_binary: bool = STL_BINARY
    coordinate_system: str = COORDINATE_SYSTEM
