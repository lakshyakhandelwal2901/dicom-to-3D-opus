"""
medrecon_engine.analysis.medgemma_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gemini API powered CT scan clinical analysis.

Uses Google's Gemini API (medgemma model or gemini-2.0-flash) to generate
an AI-driven clinical interpretation of CT scan data by analysing
representative axial slices + structured metadata.

Usage::

    from medrecon_engine.analysis.medgemma_analyzer import MedGemmaAnalyzer

    analyzer = MedGemmaAnalyzer()  # reads GEMINI_API_KEY env var
    result = analyzer.analyze(ct_arr, spacing, patient_info, tissue_stats, findings)
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image

from medrecon_engine.audit.logger import get_logger

if TYPE_CHECKING:
    from medrecon_engine.analysis.medical_findings import (
        Finding, PatientInfo, TissueStats,
    )

_log = get_logger(__name__)

# Gemini API configuration
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MODEL = "gemini-2.5-flash"


# ═══════════════════════════════════════════════════════════════════════════
# CT slice extraction
# ═══════════════════════════════════════════════════════════════════════════

def _window_ct(arr_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply CT windowing and return uint8 image."""
    lo = center - width / 2
    hi = center + width / 2
    clipped = np.clip(arr_2d, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    return scaled


def _extract_representative_slices(
    ct_arr: np.ndarray,
    n_slices: int = 5,
) -> list[Image.Image]:
    """Extract evenly-spaced axial slices with soft-tissue windowing.

    Returns PIL Images ready for Gemini input.
    """
    z_size = ct_arr.shape[0]
    # Pick slices evenly from 10% to 90% of the z-range
    indices = np.linspace(int(z_size * 0.10), int(z_size * 0.90),
                          n_slices, dtype=int)

    images: list[Image.Image] = []
    for idx in indices:
        axial = ct_arr[idx, :, :]

        # Soft-tissue window (W:400, C:40) — good for abdominal organs
        soft = _window_ct(axial, center=40, width=400)

        # Convert to RGB PIL Image
        rgb = np.stack([soft, soft, soft], axis=-1)
        img = Image.fromarray(rgb)

        # Resize to reasonable size (448x448)
        img = img.resize((448, 448), Image.LANCZOS)
        images.append(img)

    return images


def _extract_bone_slices(
    ct_arr: np.ndarray,
    n_slices: int = 3,
) -> list[Image.Image]:
    """Extract bone-windowed slices for skeletal assessment."""
    z_size = ct_arr.shape[0]
    indices = np.linspace(int(z_size * 0.2), int(z_size * 0.8),
                          n_slices, dtype=int)

    images: list[Image.Image] = []
    for idx in indices:
        axial = ct_arr[idx, :, :]
        bone = _window_ct(axial, center=400, width=1800)
        rgb = np.stack([bone, bone, bone], axis=-1)
        img = Image.fromarray(rgb)
        img = img.resize((448, 448), Image.LANCZOS)
        images.append(img)

    return images


# ═══════════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════════

def _build_analysis_prompt(
    patient: "PatientInfo",
    tissue: "TissueStats",
    findings: list["Finding"],
    spacing: tuple[float, float, float],
    volume_size: tuple[int, ...],
    n_soft: int,
    n_bone: int,
) -> str:
    """Construct the text prompt for Gemini analysis — returns structured JSON."""

    patient_block = (
        f"Patient ID: {patient.patient_id}\n"
        f"Age: {patient.patient_age}, Sex: {patient.patient_sex}, "
        f"Weight: {patient.patient_weight}\n"
        f"Study: {patient.study_description}\n"
        f"Body Part: {patient.body_part}\n"
        f"Slice Thickness: {patient.slice_thickness}\n"
        f"Contrast: {patient.contrast_agent}\n"
    )

    tissue_block = (
        f"Tissue Composition (volumes in mL):\n"
        f"  Lung: {tissue.lung_volume_ml:.0f}, Fat: {tissue.fat_volume_ml:.0f}, "
        f"Muscle: {tissue.muscle_volume_ml:.0f}\n"
        f"  Soft Tissue: {tissue.soft_tissue_volume_ml:.0f}, "
        f"Bone: {tissue.bone_volume_ml:.0f}, "
        f"Dense Bone (>700HU): {tissue.dense_bone_volume_ml:.0f}\n"
        f"  Water/Fluid: {tissue.water_fluid_volume_ml:.0f}, "
        f"Calcifications: {tissue.calcification_volume_ml:.0f}\n"
    )

    findings_block = "Automated HU-based preliminary findings:\n"
    for f in findings:
        findings_block += (
            f"  [{f.severity.upper()}] {f.organ} — {f.title}: {f.detail[:150]}"
        )
        if f.measurement:
            findings_block += f" ({f.measurement})"
        findings_block += "\n"

    vol_block = (
        f"Volume: {volume_size[0]}x{volume_size[1]}x{volume_size[2]} voxels, "
        f"Spacing: {spacing[0]:.3f}x{spacing[1]:.3f}x{spacing[2]:.3f} mm\n"
    )

    prompt = (
        f"You are an expert radiologist analyzing a CT scan. "
        f"I am providing {n_soft} soft-tissue windowed (W:400/C:40) and "
        f"{n_bone} bone-windowed (W:1800/C:400) representative axial slices "
        f"from this CT volume, along with quantitative data.\n\n"
        f"PATIENT INFORMATION:\n{patient_block}\n"
        f"SCAN DETAILS:\n{vol_block}\n"
        f"QUANTITATIVE ANALYSIS:\n{tissue_block}\n"
        f"PRELIMINARY AUTOMATED FINDINGS:\n{findings_block}\n"
        f"Analyze the images and data carefully. Return your analysis as a JSON object "
        f"with EXACTLY this structure (no markdown, no code fences, just raw JSON):\n\n"
        f'{{\n'
        f'  "findings": [\n'
        f'    {{\n'
        f'      "category": "abnormal|risk|incidental|normal",\n'
        f'      "organ": "Organ/Structure Name",\n'
        f'      "title": "Short finding title (3-8 words)",\n'
        f'      "detail": "Detailed clinical description (2-4 sentences with radiological terminology, referencing images where relevant)",\n'
        f'      "severity": "normal|mild|moderate|severe|critical",\n'
        f'      "measurement": "Optional quantitative measurement or empty string"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "summary": "2-3 sentence overall impression of the scan",\n'
        f'  "recommendations": "Clinical recommendations: follow-up imaging, lab tests, referrals (2-4 sentences)"\n'
        f'}}\n\n'
        f"RULES:\n"
        f"- Include 8-15 findings covering ALL visible organs/structures\n"
        f"- category must be one of: abnormal, risk, incidental, normal\n"
        f"- severity must be one of: normal, mild, moderate, severe, critical\n"
        f"- For normal organs, use category='normal' and severity='normal'\n"
        f"- Be specific in details — mention HU values, locations, image references\n"
        f"- Correct any false positives from the automated analysis\n"
        f"- Use standard radiological terminology\n"
        f"- Return ONLY valid JSON, no other text"
    )

    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# MedGemma Analyzer class (Gemini API)
# ═══════════════════════════════════════════════════════════════════════════

class MedGemmaAnalyzer:
    """Run medical CT analysis via Google Gemini API.

    Parameters
    ----------
    api_key : str | None
        Google Gemini API key. Falls back to GEMINI_API_KEY env var if None.
    model_name : str
        Gemini model to use (default: gemini-2.0-flash).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        **_kwargs,  # absorb legacy params like model_path
    ) -> None:
        self._api_key = api_key or DEFAULT_API_KEY
        self._model_name = model_name
        self._client = None

    def _init_client(self) -> None:
        """Initialize the Gemini API client."""
        if self._client is not None:
            return

        from google import genai

        self._client = genai.Client(api_key=self._api_key)
        _log.info("Gemini API client initialized (model=%s)", self._model_name)

    def analyze(
        self,
        ct_arr: np.ndarray,
        spacing: tuple[float, float, float],
        patient: "PatientInfo",
        tissue: "TissueStats",
        findings: list["Finding"],
        volume_size: tuple[int, ...] = (0, 0, 0),
        *,
        n_slices: int = 5,
        n_bone_slices: int = 3,
        max_new_tokens: int = 8192,
    ) -> dict:
        """Run Gemini analysis on the CT data.

        Returns
        -------
        dict with keys:
            "findings" : list[dict] — structured findings
            "summary"  : str — overall impression
            "recommendations" : str — clinical recommendations
            "raw"      : str — raw Gemini response
        """
        self._init_client()

        import json as _json
        from google.genai import types

        # Extract slices
        _log.info("Extracting %d soft-tissue + %d bone-windowed slices …",
                  n_slices, n_bone_slices)
        soft_images = _extract_representative_slices(ct_arr, n_slices=n_slices)
        bone_images = _extract_bone_slices(ct_arr, n_slices=n_bone_slices)

        # Build prompt
        _log.info("Building analysis prompt …")
        prompt_text = _build_analysis_prompt(
            patient, tissue, findings, spacing, volume_size,
            n_slices, n_bone_slices,
        )

        # Build content parts: soft-tissue images → bone images → text
        content_parts: list = []
        for img in soft_images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            content_parts.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            )
        for img in bone_images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            content_parts.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            )
        content_parts.append(types.Part.from_text(text=prompt_text))

        # Send to Gemini
        _log.info("Sending to Gemini API (%s) …", self._model_name)
        t0 = time.perf_counter()

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[types.Content(role="user", parts=content_parts)],
            config=types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )

        raw = response.text or ""
        elapsed = time.perf_counter() - t0
        _log.info("Gemini responded in %.1f s (%d chars)", elapsed, len(raw))

        # Parse JSON from response (strip markdown fences if present)
        json_str = raw.strip()
        if json_str.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            result = _json.loads(json_str)
        except _json.JSONDecodeError as exc:
            _log.warning("Failed to parse Gemini JSON: %s", exc)
            # Fallback: return raw text as single finding
            result = {
                "findings": [],
                "summary": raw[:500],
                "recommendations": "",
                "raw": raw,
            }

        # Validate findings structure
        valid_categories = {"abnormal", "risk", "incidental", "normal"}
        valid_severities = {"normal", "mild", "moderate", "severe", "critical"}
        validated_findings = []
        for f in result.get("findings", []):
            cat = f.get("category", "normal")
            sev = f.get("severity", "normal")
            if cat not in valid_categories:
                cat = "incidental"
            if sev not in valid_severities:
                sev = "moderate"
            validated_findings.append({
                "category": cat,
                "organ": f.get("organ", "General"),
                "title": f.get("title", "Finding"),
                "detail": f.get("detail", ""),
                "severity": sev,
                "measurement": f.get("measurement", ""),
            })

        result["findings"] = validated_findings
        result.setdefault("summary", "")
        result.setdefault("recommendations", "")
        result["raw"] = raw

        _log.info("Parsed %d structured findings from Gemini", len(validated_findings))
        return result

    def unload(self) -> None:
        """Clean up (no-op for API mode, kept for interface compat)."""
        self._client = None
        _log.info("Gemini API client released")
