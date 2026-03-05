"""
medrecon_engine.anatomy.registry
==================================
Factory that maps anatomy name → segmenter class.

Usage::

    segmenter = get_segmenter("bone")
    mask = segmenter.segment(hu_image)
"""

from __future__ import annotations

from medrecon_engine.config.precision_config import PrecisionConfig
from medrecon_engine.anatomy.base_segmenter import BaseSegmenter
from medrecon_engine.anatomy.bone import BoneSegmenter
from medrecon_engine.anatomy.lung import LungSegmenter
from medrecon_engine.anatomy.soft_tissue import SoftTissueSegmenter
from medrecon_engine.anatomy.brain import BrainSegmenter
from medrecon_engine.anatomy.vascular import VascularSegmenter


_REGISTRY: dict[str, type[BaseSegmenter]] = {
    "bone": BoneSegmenter,
    "lung": LungSegmenter,
    "soft_tissue": SoftTissueSegmenter,
    "brain": BrainSegmenter,
    "vascular": VascularSegmenter,
}


def get_segmenter(
    anatomy: str,
    config: PrecisionConfig | None = None,
) -> BaseSegmenter:
    """Instantiate the correct segmenter for *anatomy*.

    Raises ``ValueError`` if the anatomy is not registered.
    """
    key = anatomy.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown anatomy '{anatomy}'. Available: {available}")
    return _REGISTRY[key](config)


def list_anatomies() -> list[str]:
    """Return all registered anatomy names."""
    return sorted(_REGISTRY.keys())
