"""
medrecon_engine.visualization.colors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Organ-specific colour map for consistent visualization.

Each organ has an RGBA tuple (0-1 float) for VTK rendering
and an RGB 0-255 tuple for matplotlib overlays.
"""

from __future__ import annotations

from typing import NamedTuple


class OrganColor(NamedTuple):
    """Color definition for one organ."""
    r: float  # 0 – 1
    g: float
    b: float
    a: float  # opacity (1 = fully opaque)


# ── Organ colour palette ────────────────────────────────────────────────
ORGAN_COLORS: dict[str, OrganColor] = {
    "bones":   OrganColor(0.95, 0.92, 0.85, 1.0),   # ivory / bone white
    "lungs":   OrganColor(0.55, 0.75, 0.95, 0.35),   # light blue, translucent
    "liver":   OrganColor(0.80, 0.25, 0.20, 0.85),   # dark red
    "kidneys": OrganColor(0.85, 0.55, 0.25, 0.90),   # warm amber
}

# Matplotlib overlay (0-255 uint8, no alpha)
OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    name: (int(c.r * 255), int(c.g * 255), int(c.b * 255))
    for name, c in ORGAN_COLORS.items()
}

# Fallback colour for unknown organs
DEFAULT_COLOR = OrganColor(0.6, 0.6, 0.6, 0.7)
