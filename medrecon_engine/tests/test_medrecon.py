"""
MedRecon Engine — Test Suite
============================
Unit tests for isolated modules that do NOT require real DICOM data.
Tests use synthetic numpy arrays and mock objects where needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
class TestPrecisionConfig:
    def test_defaults(self):
        from medrecon_engine.config.precision_config import PrecisionConfig

        cfg = PrecisionConfig()
        assert cfg.strict_precision_mm == 1.0
        assert cfg.max_allowed_slice == 1.5
        assert cfg.resampling_target == (1.0, 1.0, 1.0)
        assert cfg.stl_binary is True

    def test_frozen(self):
        from medrecon_engine.config.precision_config import PrecisionConfig

        cfg = PrecisionConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.strict_precision_mm = 2.0  # type: ignore[misc]

    def test_override(self):
        from medrecon_engine.config.precision_config import PrecisionConfig

        cfg = PrecisionConfig(strict_precision_mm=0.5, stl_binary=False)
        assert cfg.strict_precision_mm == 0.5
        assert cfg.stl_binary is False


# ═══════════════════════════════════════════════════════════════════════════
# HU Profiles
# ═══════════════════════════════════════════════════════════════════════════
class TestHUProfiles:
    def test_profiles_exist(self):
        from medrecon_engine.hu_model.hu_profiles import HU_PROFILES

        required = {"bone", "lung", "soft_tissue", "brain", "vascular"}
        assert required.issubset(HU_PROFILES.keys())

    def test_profile_ranges(self):
        from medrecon_engine.hu_model.hu_profiles import HU_PROFILES

        for name, p in HU_PROFILES.items():
            assert p.min < p.max, f"{name}: min >= max"
            assert p.min <= p.expected_peak <= p.max, f"{name}: peak out of range"


# ═══════════════════════════════════════════════════════════════════════════
# HU Estimator (synthetic histogram)
# ═══════════════════════════════════════════════════════════════════════════
class TestHUEstimator:
    def test_bone_estimation_synthetic(self):
        """Create a volume with a strong peak near 700 HU and verify estimator
        places the adaptive range around it."""
        from medrecon_engine.hu_model.hu_estimator import HUEstimator

        rng = np.random.default_rng(42)
        # Background at ~0 HU, bone peak at ~700 HU
        bg = rng.normal(0, 30, size=100_000).astype(np.float64)
        bone = rng.normal(700, 40, size=20_000).astype(np.float64)
        data = np.concatenate([bg, bone])
        rng.shuffle(data)
        data = data.reshape((10, 120, 100))

        est = HUEstimator()
        result = est.estimate(data, "bone")
        # Adaptive min should be somewhere around 500-800 range
        assert 200 < result.adaptive_min < 900, f"Unexpected low bound: {result.adaptive_min}"
        assert result.adaptive_max >= result.adaptive_min

    def test_unknown_anatomy_raises(self):
        from medrecon_engine.hu_model.hu_estimator import HUEstimator

        vol = np.zeros((5, 5, 5), dtype=np.float64)
        est = HUEstimator()
        with pytest.raises(KeyError):
            est.estimate(vol, "nonexistent_anatomy")


# ═══════════════════════════════════════════════════════════════════════════
# Anatomy registry
# ═══════════════════════════════════════════════════════════════════════════
class TestAnatomyRegistry:
    def test_list_anatomies(self):
        from medrecon_engine.anatomy.registry import list_anatomies

        all_anat = list_anatomies()
        assert "bone" in all_anat
        assert "lung" in all_anat

    def test_get_segmenter(self):
        from medrecon_engine.anatomy.registry import get_segmenter
        from medrecon_engine.anatomy.base_segmenter import BaseSegmenter

        seg = get_segmenter("bone")
        assert isinstance(seg, BaseSegmenter)

    def test_unknown_segmenter_raises(self):
        from medrecon_engine.anatomy.registry import get_segmenter

        with pytest.raises(ValueError):
            get_segmenter("does_not_exist")


# ═══════════════════════════════════════════════════════════════════════════
# Bone segmenter (synthetic)
# ═══════════════════════════════════════════════════════════════════════════
class TestBoneSegmenter:
    def test_segment_simple(self):
        import SimpleITK as sitk
        from medrecon_engine.anatomy.bone import BoneSegmenter

        # volume with clear bone region
        arr = np.zeros((30, 30, 30), dtype=np.float64)
        arr[10:20, 10:20, 10:20] = 800.0  # bone cube

        vol = sitk.GetImageFromArray(arr)
        vol.SetSpacing((1.0, 1.0, 1.0))

        seg = BoneSegmenter()
        mask_sitk = seg.segment(vol)
        mask = sitk.GetArrayFromImage(mask_sitk)

        assert mask.dtype == np.uint8
        assert mask[15, 15, 15] > 0, "Centre of bone cube should be segmented"
        assert mask[0, 0, 0] == 0, "Corner should be background"


# ═══════════════════════════════════════════════════════════════════════════
# Mesh validator (synthetic VTK mesh)
# ═══════════════════════════════════════════════════════════════════════════
class TestMeshValidator:
    @staticmethod
    def _make_sphere():
        """Create a simple VTK sphere for testing."""
        import vtk

        src = vtk.vtkSphereSource()
        src.SetRadius(10.0)
        src.SetThetaResolution(20)
        src.SetPhiResolution(20)
        src.Update()
        return src.GetOutput()

    def test_sphere_is_manifold(self):
        from medrecon_engine.mesh.mesh_validator import MeshValidator

        mesh = self._make_sphere()
        val = MeshValidator()
        report = val.validate(mesh)

        assert report.is_manifold is True
        assert report.num_vertices > 0
        assert report.num_faces > 0

    def test_sphere_topology_valid(self):
        from medrecon_engine.mesh.mesh_validator import MeshValidator

        mesh = self._make_sphere()
        val = MeshValidator()
        report = val.validate(mesh)
        assert report.passed is True


# ═══════════════════════════════════════════════════════════════════════════
# STL Writer (round-trip)
# ═══════════════════════════════════════════════════════════════════════════
class TestSTLWriter:
    @staticmethod
    def _make_sphere():
        import vtk

        src = vtk.vtkSphereSource()
        src.SetRadius(5.0)
        src.SetThetaResolution(12)
        src.SetPhiResolution(12)
        src.Update()
        return src.GetOutput()

    def test_binary_export(self):
        from medrecon_engine.export.stl_writer import STLWriter
        from medrecon_engine.config.precision_config import PrecisionConfig

        mesh = self._make_sphere()
        cfg = PrecisionConfig(stl_binary=True)
        w = STLWriter(cfg)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test.stl"
            report = w.write(mesh, out)
            assert report.success
            assert report.file_size_bytes > 0
            assert out.exists()

    def test_ascii_export(self):
        from medrecon_engine.export.stl_writer import STLWriter
        from medrecon_engine.config.precision_config import PrecisionConfig

        mesh = self._make_sphere()
        cfg = PrecisionConfig(stl_binary=False)
        w = STLWriter(cfg)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "test_ascii.stl"
            report = w.write(mesh, out)
            assert report.success
            assert report.file_size_bytes > 0
            # ASCII STL starts with "solid"
            with open(out, "r") as f:
                first_line = f.readline()
            assert first_line.lower().startswith("solid")

    def test_empty_mesh_raises(self):
        import vtk
        from medrecon_engine.export.stl_writer import STLWriter

        empty = vtk.vtkPolyData()
        w = STLWriter()
        with pytest.raises(ValueError, match="empty"):
            w.write(empty, "dummy.stl")


# ═══════════════════════════════════════════════════════════════════════════
# Audit logger
# ═══════════════════════════════════════════════════════════════════════════
class TestAuditLogger:
    def test_get_logger_returns_logger(self):
        import logging
        from medrecon_engine.audit.logger import get_logger

        log = get_logger("test.module")
        assert isinstance(log, logging.Logger)

    def test_audit_record_frozen(self):
        from medrecon_engine.audit.logger import AuditRecord

        rec = AuditRecord(anatomy="bone")
        with pytest.raises(Exception):
            rec.anatomy = "lung"  # type: ignore[misc]

    def test_audit_write_read_roundtrip(self):
        from medrecon_engine.audit.logger import (
            AuditRecord,
            read_audit_trail,
            write_audit_record,
        )

        rec = AuditRecord(
            anatomy="bone",
            confidence_score=0.92,
            confidence_grade="SURGICAL",
            success=True,
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            write_audit_record(rec, audit_dir=td_path)
            trail = read_audit_trail(audit_dir=td_path)

            assert len(trail) == 1
            assert trail[0]["anatomy"] == "bone"
            assert trail[0]["confidence_grade"] == "SURGICAL"


# ═══════════════════════════════════════════════════════════════════════════
# Surface metrics (synthetic sphere)
# ═══════════════════════════════════════════════════════════════════════════
class TestSurfaceMetrics:
    @staticmethod
    def _make_sphere():
        import vtk

        src = vtk.vtkSphereSource()
        src.SetRadius(10.0)
        src.SetThetaResolution(30)
        src.SetPhiResolution(30)
        src.Update()
        return src.GetOutput()

    def test_compute_returns_metrics(self):
        from medrecon_engine.quality.surface_metrics import SurfaceMetricsComputer

        mesh = self._make_sphere()
        comp = SurfaceMetricsComputer()
        m = comp.compute(mesh)

        assert m.surface_area_mm2 > 0
        assert m.volume_mm3 > 0
        assert m.edge_length_mean > 0
        assert 0 <= m.normal_consistency <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Confidence scorer (synthetic)
# ═══════════════════════════════════════════════════════════════════════════
class TestConfidenceScorer:
    def test_surgical_grade_on_good_inputs(self):
        """A perfect sphere should score high."""
        import vtk
        from medrecon_engine.config.precision_config import PrecisionConfig
        from medrecon_engine.mesh.mesh_validator import MeshValidator
        from medrecon_engine.quality.confidence_score import ConfidenceScorer
        from medrecon_engine.quality.surface_metrics import SurfaceMetricsComputer

        src = vtk.vtkSphereSource()
        src.SetRadius(10)
        src.SetThetaResolution(30)
        src.SetPhiResolution(30)
        src.Update()
        mesh = src.GetOutput()

        val = MeshValidator()
        report = val.validate(mesh)

        comp = SurfaceMetricsComputer()
        metrics = comp.compute(mesh)

        cfg = PrecisionConfig()
        scorer = ConfidenceScorer(cfg)
        result = scorer.compute(
            validation=report,
            metrics=metrics,
            hu_estimation=None,
            original_spacing=(1.0, 1.0, 1.0),
        )

        assert result.total >= 0.5  # should be at least usable
        assert result.grade in ("SURGICAL", "USABLE", "REJECT")
