"""
medrecon_engine.analysis.threshold_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Derive **adaptive, scan-specific HU thresholds** from the per-tissue
statistics computed by :mod:`hu_analyzer`.

The key insight: instead of hard-coded thresholds like ``bone > 200 HU``,
we compute them from the actual tissue statistics of each scan.

Default rule::

    threshold_low  = mean - k * std      (clamped to p5)
    threshold_high = mean + k * std      (clamped to p95)

Where **k** defaults to 2.0 (covers ~95 % of the Gaussian distribution).

Some tissues use custom rules:

* **bone** — high-end only (we only care about the lower boundary,
  since high-HU voxels are always bone).
* **lung** — inverted: lung tissue is strongly negative HU.
* **soft tissues** (liver, kidney, spleen, heart) — symmetric ±k·σ.

The output is a ``dict[str, tuple[float, float]]`` mapping tissue name
to ``(low_hu, high_hu)`` — ready to feed into
:mod:`adaptive_segmenter`.
"""

from __future__ import annotations

from medrecon_engine.audit.logger import get_logger

_log = get_logger(__name__)

# ── Default safety clamps ─────────────────────────────────────────────────
# Even with adaptive thresholds, we don't want insane ranges.

_CLAMPS: dict[str, tuple[float, float]] = {
    "bone":   (100.0, 3000.0),
    "liver":  (-30.0, 200.0),
    "kidney": (-30.0, 200.0),
    "spleen": (-30.0, 200.0),
    "lung":   (-1000.0, -200.0),
    "heart":  (-30.0, 200.0),
    "aorta":  (-30.0, 400.0),
    "muscle": (-50.0, 150.0),
    "fat":    (-200.0, -20.0),
}

# Fallback thresholds when no AI data is available
FALLBACK_THRESHOLDS: dict[str, tuple[float, float]] = {
    "bone":   (200.0, 2000.0),
    "liver":  (40.0, 80.0),
    "kidney": (25.0, 65.0),
    "spleen": (35.0, 75.0),
    "lung":   (-900.0, -400.0),
    "heart":  (30.0, 80.0),
    "aorta":  (100.0, 300.0),
    "muscle": (10.0, 80.0),
    "fat":    (-150.0, -50.0),
}


def derive_thresholds(
    profiles: dict[str, dict[str, float]],
    *,
    k: float = 2.0,
) -> dict[str, tuple[float, float]]:
    """Convert per-tissue HU profiles into adaptive (low, high) ranges.

    Parameters
    ----------
    profiles : dict[str, dict]
        Output of ``hu_analyzer.analyze_all_labels()``.  Each value has
        keys: ``mean``, ``std``, ``p5``, ``p95``, etc.
    k : float
        Number of standard deviations for range expansion.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``{"bone": (210.0, 1950.0), "liver": (44.0, 76.0), ...}``
    """
    thresholds: dict[str, tuple[float, float]] = {}

    for tissue, stats in profiles.items():
        mean = stats["mean"]
        std  = stats["std"]
        p5   = stats["p5"]
        p95  = stats["p95"]

        # Default: mean ± k * std, clamped to p5/p95 for robustness
        raw_low  = mean - k * std
        raw_high = mean + k * std

        # Use the more conservative of statistical & percentile bounds
        low  = max(raw_low, p5)
        high = min(raw_high, p95)

        # Ensure low < high
        if low >= high:
            low  = p5
            high = p95

        # Apply absolute safety clamps
        clamp = _CLAMPS.get(tissue, (-1500.0, 3500.0))
        low  = max(low, clamp[0])
        high = min(high, clamp[1])

        # Bone special case: we care most about the lower bound
        # (everything above is bone); expand the upper end generously
        if tissue == "bone":
            high = max(high, 2000.0)

        thresholds[tissue] = (round(low, 1), round(high, 1))

        _log.info(
            "  %-10s  threshold: [%7.1f … %7.1f]  (mean=%.1f std=%.1f)",
            tissue,
            low,
            high,
            mean,
            std,
        )

    # Fill in fallbacks for any missing tissues
    for tissue, fb in FALLBACK_THRESHOLDS.items():
        if tissue not in thresholds:
            thresholds[tissue] = fb
            _log.info("  %-10s  fallback:  [%7.1f … %7.1f]", tissue, fb[0], fb[1])

    return thresholds


def thresholds_summary(thresholds: dict[str, tuple[float, float]]) -> str:
    """Return a human-readable summary table of adaptive thresholds."""
    lines = ["Adaptive HU Thresholds", "=" * 45]
    for tissue, (lo, hi) in sorted(thresholds.items()):
        lines.append(f"  {tissue:12s}  {lo:8.1f}  –  {hi:8.1f} HU")
    return "\n".join(lines)
