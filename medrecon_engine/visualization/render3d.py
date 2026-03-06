"""
medrecon_engine.visualization.render3d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interactive 3D rendering of segmented meshes using VTK.

Features
--------
- Organ-specific colours + opacity
- Offscreen screenshot export (no display required)
- Interactive window (when display available)
- Multi-view (front / side / top) screenshot strip
- Automatic camera framing around all meshes

Usage::

    from medrecon_engine.visualization.render3d import render_meshes, save_screenshot

    # Interactive
    render_meshes(vtk_meshes)

    # Offscreen screenshots
    save_screenshot(vtk_meshes, output_dir / "preview.png")
    save_multi_view(vtk_meshes, output_dir / "views.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import vtk

from medrecon_engine.audit.logger import get_logger
from medrecon_engine.visualization.colors import ORGAN_COLORS, DEFAULT_COLOR

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_actor(mesh: vtk.vtkPolyData, organ: str) -> vtk.vtkActor:
    """Create a VTK actor for one organ mesh with proper colour/opacity."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.ScalarVisibilityOff()

    color = ORGAN_COLORS.get(organ, DEFAULT_COLOR)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color.r, color.g, color.b)
    actor.GetProperty().SetOpacity(color.a)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetAmbient(0.2)
    actor.GetProperty().SetDiffuse(0.8)

    return actor


def _build_renderer(
    vtk_meshes: dict[str, vtk.vtkPolyData],
    bg_color: tuple[float, float, float] = (0.1, 0.1, 0.15),
) -> vtk.vtkRenderer:
    """Build a VTK renderer with actors for all organ meshes."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*bg_color)

    for organ, mesh in vtk_meshes.items():
        if mesh.GetNumberOfPoints() == 0:
            continue
        actor = _build_actor(mesh, organ)
        renderer.AddActor(actor)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Elevation(15)
    cam.Azimuth(30)
    renderer.ResetCameraClippingRange()

    return renderer


def _add_legend(renderer: vtk.vtkRenderer, organs: list[str]) -> None:
    """Add a colour legend to the renderer."""
    legend = vtk.vtkLegendBoxActor()
    legend.SetNumberOfEntries(len(organs))

    for i, organ in enumerate(organs):
        color = ORGAN_COLORS.get(organ, DEFAULT_COLOR)
        # Create a small sphere icon
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.5)
        sphere.Update()
        legend.SetEntry(i, sphere.GetOutput(), organ.capitalize(),
                        [color.r, color.g, color.b])

    legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    legend.GetPositionCoordinate().SetValue(0.01, 0.01)
    legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
    legend.GetPosition2Coordinate().SetValue(0.18, 0.20)
    legend.UseBackgroundOn()
    legend.SetBackgroundColor(0.15, 0.15, 0.2)
    legend.SetBackgroundOpacity(0.7)

    renderer.AddActor(legend)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def render_meshes(
    vtk_meshes: dict[str, vtk.vtkPolyData],
    *,
    window_size: tuple[int, int] = (1280, 800),
    title: str = "MedRecon — 3D Segmentation Viewer",
) -> None:
    """Open an interactive VTK window showing all organ meshes.

    Parameters
    ----------
    vtk_meshes : dict mapping organ name → vtkPolyData
    window_size : (width, height) in pixels
    title : Window title
    """
    _log.info("Opening interactive 3D viewer (%d meshes)", len(vtk_meshes))

    renderer = _build_renderer(vtk_meshes)
    _add_legend(renderer, list(vtk_meshes.keys()))

    # Axes indicator
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)

    window = vtk.vtkRenderWindow()
    window.SetSize(*window_size)
    window.SetWindowName(title)
    window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.15, 0.15)
    widget.EnabledOn()
    widget.InteractiveOff()

    # Add text overlay
    text = vtk.vtkTextActor()
    text.SetInput(title)
    text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text.GetPositionCoordinate().SetValue(0.02, 0.95)
    text.GetTextProperty().SetFontSize(18)
    text.GetTextProperty().SetColor(0.9, 0.9, 0.9)
    text.GetTextProperty().SetFontFamilyToCourier()
    renderer.AddActor2D(text)

    window.Render()
    interactor.Start()


def save_screenshot(
    vtk_meshes: dict[str, vtk.vtkPolyData],
    output_path: str | Path,
    *,
    size: tuple[int, int] = (1920, 1080),
    bg_color: tuple[float, float, float] = (0.1, 0.1, 0.15),
) -> Path:
    """Render offscreen and save a PNG screenshot.

    Works on headless machines (no display needed).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = _build_renderer(vtk_meshes, bg_color=bg_color)
    _add_legend(renderer, list(vtk_meshes.keys()))

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*size)
    window.AddRenderer(renderer)
    window.Render()

    win2img = vtk.vtkWindowToImageFilter()
    win2img.SetInput(window)
    win2img.SetInputBufferTypeToRGBA()
    win2img.ReadFrontBufferOff()
    win2img.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(win2img.GetOutputPort())
    writer.Write()

    _log.info("Screenshot saved: %s", output_path)
    return output_path


def save_multi_view(
    vtk_meshes: dict[str, vtk.vtkPolyData],
    output_path: str | Path,
    *,
    size: tuple[int, int] = (2400, 800),
    bg_color: tuple[float, float, float] = (0.1, 0.1, 0.15),
) -> Path:
    """Render 3 views (front, right side, top) side-by-side as one PNG.

    Returns the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    views = [
        ("Front",  0,   0),   # (elevation, azimuth)
        ("Right", 0,  90),
        ("Top",   90,  0),
    ]

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*size)

    renderers = []
    w = 1.0 / len(views)

    for i, (label, elev, azim) in enumerate(views):
        renderer = _build_renderer(vtk_meshes, bg_color=bg_color)

        # Set viewport for this tile
        renderer.SetViewport(i * w, 0.0, (i + 1) * w, 1.0)

        # Reposition camera
        cam = renderer.GetActiveCamera()
        cam.SetPosition(0, 0, 1)
        cam.SetViewUp(0, 1, 0)
        renderer.ResetCamera()
        cam.Elevation(elev)
        cam.Azimuth(azim)
        renderer.ResetCameraClippingRange()

        # View label
        text = vtk.vtkTextActor()
        text.SetInput(label)
        text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text.GetPositionCoordinate().SetValue(0.02, 0.92)
        text.GetTextProperty().SetFontSize(16)
        text.GetTextProperty().SetColor(0.9, 0.9, 0.9)
        renderer.AddActor2D(text)

        window.AddRenderer(renderer)
        renderers.append(renderer)

    window.Render()

    win2img = vtk.vtkWindowToImageFilter()
    win2img.SetInput(window)
    win2img.SetInputBufferTypeToRGBA()
    win2img.ReadFrontBufferOff()
    win2img.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(win2img.GetOutputPort())
    writer.Write()

    _log.info("Multi-view saved: %s", output_path)
    return output_path
