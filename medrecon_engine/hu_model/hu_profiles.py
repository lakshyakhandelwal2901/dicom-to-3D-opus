"""
medrecon_engine.hu_model.hu_profiles
======================================
Static, literature-derived HU profiles for every supported anatomy.

Each profile specifies:
- ``expected_peak``: the dominant HU value for this tissue in a typical
  non-contrast CT.  Used as the histogram anchor.
- ``min`` / ``max``: absolute hard limits — the adaptive estimator will
  never go outside this range no matter what the histogram says.
- ``contrast_shift``: expected shift when IV contrast is present (positive
  means the peak moves UP).  Used in future contrast-detection logic.

Sources:
  Bushberg, "The Essential Physics of Medical Imaging", 4th ed.
  Prokop & Galanski, "Spiral and Multislice CT", 2nd ed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HUProfile:
    """Immutable HU range profile for a single anatomy."""

    expected_peak: float    # typical dominant HU
    min: float              # hard lower bound
    max: float              # hard upper bound
    contrast_shift: float = 0.0   # expected shift with contrast agent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Profile registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

HU_PROFILES: dict[str, HUProfile] = {

    # ── Bone (cortical + cancellous) ─────────────────────────────────── #
    "bone": HUProfile(
        expected_peak=700,
        min=200,
        max=3000,
        contrast_shift=0,    # bone is unaffected by contrast
    ),

    # ── Lung parenchyma ─────────────────────────────────────────────── #
    "lung": HUProfile(
        expected_peak=-700,
        min=-1000,
        max=-300,
        contrast_shift=0,
    ),

    # ── Soft tissue (muscle, connective) ─────────────────────────────── #
    "soft_tissue": HUProfile(
        expected_peak=50,
        min=-100,
        max=200,
        contrast_shift=30,   # slight enhancement with contrast
    ),

    # ── Brain grey + white matter ─────────────────────────────────────── #
    "brain": HUProfile(
        expected_peak=35,
        min=-20,
        max=80,
        contrast_shift=20,
    ),

    # ── Vascular (aorta, major vessels, post-contrast) ───────────────── #
    "vascular": HUProfile(
        expected_peak=250,
        min=100,
        max=500,
        contrast_shift=150,  # large shift — contrast pools in vessels
    ),

    # ── Liver parenchyma ─────────────────────────────────────────────── #
    "liver": HUProfile(
        expected_peak=60,
        min=30,
        max=180,
        contrast_shift=40,
    ),

    # ── Kidney cortex ────────────────────────────────────────────────── #
    "kidney": HUProfile(
        expected_peak=30,
        min=10,
        max=200,
        contrast_shift=80,
    ),

    # ── Fat ──────────────────────────────────────────────────────────── #
    "fat": HUProfile(
        expected_peak=-80,
        min=-200,
        max=-30,
        contrast_shift=0,
    ),

    # ── Cartilage ────────────────────────────────────────────────────── #
    "cartilage": HUProfile(
        expected_peak=120,
        min=80,
        max=200,
        contrast_shift=10,
    ),

    # ── Metal implant ────────────────────────────────────────────────── #
    "metal_implant": HUProfile(
        expected_peak=2500,
        min=1000,
        max=3071,
        contrast_shift=0,
    ),
}


def get_profile(anatomy: str) -> HUProfile:
    """Retrieve a profile by anatomy name.  Raises ``KeyError`` if unknown."""
    key = anatomy.lower().strip()
    if key not in HU_PROFILES:
        available = ", ".join(sorted(HU_PROFILES.keys()))
        raise KeyError(f"Unknown anatomy '{anatomy}'. Available: {available}")
    return HU_PROFILES[key]
