"""
medrecon_engine.config.hu_ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Central Hounsfield Unit (HU) threshold configuration.

Each structure has a (low, high) HU range, an output filename,
and an output folder category.  Organ-specific segmentation logic
lives in :pymod:`anatomy.hu_segmenter` — this file is just the
single source of truth for HU values.

HU values are physically meaningful and stable across CT scanners,
making this a deterministic, reproducible segmentation system.
"""

from __future__ import annotations

from typing import NamedTuple


class HUSpec(NamedTuple):
    """Specification for a single structure."""
    hu_low: float      # Lower HU threshold (inclusive)
    hu_high: float     # Upper HU threshold (inclusive)
    filename: str      # Output OBJ filename (without extension)
    category: str      # Output folder: bones, organs, lungs, others


# ═══════════════════════════════════════════════════════════════════════════
# HU_RANGES — single source of truth
# ═══════════════════════════════════════════════════════════════════════════
# Processing order matters — bones first, then lungs, then soft tissue.
# The segmenter uses hierarchical subtraction so higher-priority masks
# are removed from lower-priority ones.

HU_RANGES: dict[str, HUSpec] = {
    # ── Skeleton ──────────────────────────────────────────────────────
    # 300 HU lower bound excludes intestinal contrast (200-300 HU)
    "bones":           HUSpec( 300,  3000, filename="bones",   category="bones"),

    # ── Lungs ─────────────────────────────────────────────────────────
    "lungs":           HUSpec(-950,  -650, filename="lungs",   category="lungs"),

    # ── Soft-tissue organs ────────────────────────────────────────────
    "liver":           HUSpec(  30,    80, filename="liver",   category="organs"),
    "kidneys":         HUSpec(  20,    45, filename="kidneys", category="organs"),
}

# Ordered list — segmentation priority (first wins overlapping voxels)
STRUCTURE_ORDER: list[str] = ["bones", "lungs", "liver", "kidneys"]

