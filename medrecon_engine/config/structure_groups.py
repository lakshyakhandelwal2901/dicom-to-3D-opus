"""
medrecon_engine.config.structure_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maps TotalSegmentator label names to anatomical categories and
defines which labels should be merged into combined organs.

Categories
----------
* **bones**   — skeletal structures (vertebrae, ribs, pelvis, femur, …)
* **organs**  — abdominal / thoracic organs (liver, kidneys, spleen, …)
* **lungs**   — pulmonary structures
* **vessels** — major blood vessels (aorta, vena cava, iliac, …)
* **muscles** — named muscle groups
* **others**  — anything not matched above

Merge Rules
-----------
Labels sharing a merge key are combined into a single mesh.
E.g. all ``vertebrae_*`` → ``spine.obj``, all ``rib_left_*`` + ``rib_right_*``
→ ``ribs.obj``.
"""

from __future__ import annotations

# ── Category prefixes ─────────────────────────────────────────────────────
# If a label *contains* any of these substrings it is routed to that group.

BONES: list[str] = [
    "vertebrae",
    "spine",
    "rib",
    "ribs",
    "sacrum",
    "hip_",
    "pelvis",
    "femur",
    "scapula",
    "humerus",
    "clavicula",
    "sternum",
    "costal_cartilage",
    "skull",
]

ORGANS: list[str] = [
    "liver",
    "kidney",
    "kidneys",
    "spleen",
    "pancreas",
    "heart",
    "gallbladder",
    "stomach",
    "duodenum",
    "colon",
    "small_bowel",
    "esophagus",
    "adrenal_gland",
    "adrenals",
    "urinary_bladder",
    "prostate",
    "uterus",
    "rectum",
    "brain",
    "thyroid_gland",
]

LUNGS: list[str] = [
    "lung",
    "lungs",
    "trachea",
]

VESSELS: list[str] = [
    "aorta",
    "vena_cava",
    "inferior_vena_cava",
    "superior_vena_cava",
    "portal_vein",
    "splenic_vein",
    "iliac_artery",
    "iliac_arteries",
    "iliac_vena",
    "iliac_veins",
    "pulmonary_artery",
    "pulmonary_vein",
    "celiac_trunk",
    "superior_mesenteric_artery",
    "hepatic_vein",
    "brachiocephalic",
    "subclavian_artery",
    "common_carotid",
]

MUSCLES: list[str] = [
    "gluteus",
    "iliopsoas",
    "autochthon",
    "erector_spinae",
    "spinal_cord",
]


def get_category(label_name: str) -> str:
    """Return the folder category for a TotalSegmentator label name.

    Parameters
    ----------
    label_name : str
        E.g. ``"vertebrae_L3"``, ``"liver"``, ``"aorta"``.

    Returns
    -------
    str
        One of ``"bones"``, ``"organs"``, ``"lungs"``, ``"vessels"``,
        ``"muscles"``, ``"others"``.
    """
    ln = label_name.lower()

    for key in BONES:
        if key in ln:
            return "bones"

    for key in ORGANS:
        if key in ln:
            return "organs"

    for key in LUNGS:
        if key in ln:
            return "lungs"

    for key in VESSELS:
        if key in ln:
            return "vessels"

    for key in MUSCLES:
        if key in ln:
            return "muscles"

    return "others"


# ── Merge rules ───────────────────────────────────────────────────────────
# Key = output file stem, Value = list of prefixes that get merged.
# Labels whose name starts with any listed prefix are combined.

MERGE_RULES: dict[str, list[str]] = {
    # Bones
    "spine":       ["vertebrae_"],
    "ribs":        ["rib_left_", "rib_right_", "rib_"],
    "pelvis":      ["hip_left", "hip_right", "sacrum"],
    "sternum":     ["sternum", "costal_cartilage"],

    # Organs
    "kidneys":     ["kidney_left", "kidney_right"],
    "adrenals":    ["adrenal_gland_left", "adrenal_gland_right"],

    # Lungs
    "lungs":       ["lung_"],

    # Vessels
    "iliac_arteries": ["iliac_artery_left", "iliac_artery_right"],
    "iliac_veins":    ["iliac_vena_left", "iliac_vena_right"],

    # Muscles
    "gluteus":        ["gluteus_"],
    "iliopsoas":      ["iliopsoas_left", "iliopsoas_right"],
    "autochthon":     ["autochthon_left", "autochthon_right"],
}


def get_merge_target(label_name: str) -> str | None:
    """Return the merge target name if this label should be merged.

    Parameters
    ----------
    label_name : str
        E.g. ``"vertebrae_L1"``.

    Returns
    -------
    str | None
        Merge target (e.g. ``"spine"``) or *None* if the label stands alone.
    """
    ln = label_name.lower()
    for target, prefixes in MERGE_RULES.items():
        for prefix in prefixes:
            if ln.startswith(prefix) or ln == prefix:
                return target
    return None
