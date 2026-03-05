# medrecon_engine/mesh/vtk_generator.py

import vtk
import SimpleITK as sitk
import numpy as np


def generate_mesh(mask_image):
    """Convert a SimpleITK binary mask to a VTK surface mesh.

    Uses ``vtkImageImport`` — the correct, medical-grade bridge from
    numpy volumes to VTK.  No manual transpose, no flatten-order
    confusion, no extent mismatches.

    Parameters
    ----------
    mask_image : sitk.Image
        Binary mask (0 / 1) with physical spacing & origin set.

    Returns
    -------
    vtk.vtkPolyData
        Raw iso-surface from marching cubes.
    """

    # Get numpy array from SimpleITK (z, y, x)
    array = sitk.GetArrayFromImage(mask_image).astype(np.uint8)

    depth, height, width = array.shape

    spacing = mask_image.GetSpacing()
    origin = mask_image.GetOrigin()

    # Debug: report non-zero voxels before mesh extraction
    nz = np.count_nonzero(array)
    print(f"Mask non-zero voxels: {nz}")

    # Use vtkImageImport (correct method)
    importer = vtk.vtkImageImport()

    data_string = array.tobytes()

    importer.CopyImportVoidPointer(data_string, len(data_string))
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(1)

    importer.SetDataExtent(0, width - 1,
                           0, height - 1,
                           0, depth - 1)

    importer.SetWholeExtent(0, width - 1,
                            0, height - 1,
                            0, depth - 1)

    importer.SetDataSpacing(spacing)
    importer.SetDataOrigin(origin)

    importer.Update()

    # Flying Edges — faster than marching cubes, smoother surfaces,
    # better topology.  Same interface, drop-in replacement.
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputConnection(importer.GetOutputPort())
    fe.SetValue(0, 0.5)
    fe.ComputeNormalsOn()
    fe.Update()

    return fe.GetOutput()


def _sitk_to_vtk(image: sitk.Image, scalar_type: str = "float") -> vtk.vtkImageData:
    """Bridge any SimpleITK image → vtkImageData via numpy_to_vtk.

    Uses ``vtk.util.numpy_support.numpy_to_vtk`` + ``vtkImageData``
    instead of ``vtkImageImport.CopyImportVoidPointer`` which can
    segfault on larger volumes.

    Parameters
    ----------
    image : sitk.Image
        Input volume (any pixel type — will be cast internally).
    scalar_type : str
        ``"float"`` (default) or ``"uint8"``.

    Returns
    -------
    vtk.vtkImageData
    """
    from vtk.util.numpy_support import numpy_to_vtk

    if scalar_type == "uint8":
        arr = sitk.GetArrayFromImage(image).astype(np.uint8)
        vtk_type = vtk.VTK_UNSIGNED_CHAR
    else:
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        vtk_type = vtk.VTK_FLOAT

    depth, height, width = arr.shape
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    flat = arr.ravel(order="C")
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk_type)

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(width, height, depth)
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(origin)
    vtk_img.GetPointData().SetScalars(vtk_arr)

    return vtk_img


def generate_mesh_gradient_guided(
    ct_image: sitk.Image,
    mask_image: sitk.Image,
    *,
    gauss_variance: float = 1.2,
    iso_value: float = 0.1,
) -> vtk.vtkPolyData:
    """Gradient-guided surface extraction for sub-voxel accuracy.

    Instead of extracting an iso-surface from a binary mask (which
    produces voxel stair-steps), this function uses the **original CT
    intensity gradients** restricted to the segmented region.

    Pipeline::

        CT → Gaussian smooth → GradientMagnitude → mask with label
        → FlyingEdges3D on gradient field

    This preserves fine anatomical detail that binary masks destroy.

    Parameters
    ----------
    ct_image : sitk.Image
        Original CT volume in Hounsfield Units (full resolution —
        do **not** downsample).
    mask_image : sitk.Image
        Binary organ mask (0/1) from TotalSegmentator.
    gauss_variance : float
        Gaussian smoothing variance (mm²).  1.0–1.5 is typical.
    iso_value : float
        Iso-surface threshold on the masked gradient field.
        Lower = captures more detail; higher = cleaner but loses fine
        edges.  0.1 is a good default for most organs.

    Returns
    -------
    vtk.vtkPolyData
        Raw iso-surface extracted from gradient boundaries.
    """
    # 1 — Smooth CT slightly to suppress noise while preserving edges
    smoothed = sitk.DiscreteGaussian(sitk.Cast(ct_image, sitk.sitkFloat32),
                                     variance=gauss_variance)

    # 2 — Compute gradient magnitude (edge detector)
    gradient = sitk.GradientMagnitude(smoothed)

    # 3 — Restrict gradient to the segmented region only
    mask_f = sitk.Cast(mask_image, sitk.sitkFloat32)

    # Resample mask to gradient space if sizes differ
    # (TotalSegmentator may output at a different resolution)
    if gradient.GetSize() != mask_f.GetSize():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(gradient)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_f = resampler.Execute(mask_f)

    refined = gradient * mask_f  # zero outside the organ

    # 4 — Convert to VTK and extract iso-surface
    vtk_img = _sitk_to_vtk(refined, scalar_type="float")

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(vtk_img)
    fe.SetValue(0, iso_value)
    fe.ComputeNormalsOn()
    fe.Update()

    return fe.GetOutput()


def improve_mesh(
    mesh: vtk.vtkPolyData,
    *,
    hole_size: float = 1000.0,
    smooth_iterations: int = 20,
    smooth_passband: float = 0.1,
    target_reduction: float = 0.5,
) -> vtk.vtkPolyData:
    """Post-process a raw iso-surface for surgical-quality output.

    Pipeline::

        FillHoles → WindowedSinc smooth → QuadricDecimation → Normals

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Raw mesh from :func:`generate_mesh`.
    hole_size : float
        Maximum hole perimeter to fill (VTK units).
    smooth_iterations : int
        WindowedSinc smoothing passes.
    smooth_passband : float
        Smoothing passband (lower = more aggressive).
    target_reduction : float
        Fraction of faces to *remove* (0.5 = keep 50%).

    Returns
    -------
    vtk.vtkPolyData
        Cleaned, smoothed, decimated mesh with consistent normals.
    """
    # 1 — Fill holes
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(mesh)
    filler.SetHoleSize(hole_size)
    filler.Update()
    mesh = filler.GetOutput()

    # 2 — Windowed Sinc smoothing (Taubin — no shrinkage)
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh)
    smoother.SetNumberOfIterations(smooth_iterations)
    smoother.SetPassBand(smooth_passband)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    mesh = smoother.GetOutput()

    # 3 — Quadric decimation
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(mesh)
    decimator.SetTargetReduction(target_reduction)
    decimator.VolumePreservationOn()
    decimator.Update()
    mesh = decimator.GetOutput()

    # 4 — Recompute consistent outward normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def _downsample_sitk(image: sitk.Image, factor: int = 2) -> sitk.Image:
    """Resample a SimpleITK image to coarser spacing (factor × original).

    Uses linear interpolation for float images and nearest-neighbour
    for integer/label images.
    """
    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()
    new_spacing = [s * factor for s in orig_spacing]
    new_size = [max(1, int(round(sz * sp / nsp)))
                for sz, sp, nsp in zip(orig_size, orig_spacing, new_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputPixelType(image.GetPixelID())

    ptype = image.GetPixelIDTypeAsString()
    if "float" in ptype.lower():
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    return resampler.Execute(image)


# ─── Voxel-count thresholds for downsampling ──────────────────────────
_LARGE_CROP_VOXELS = 5_000_000   # 2× downsample above this
_HUGE_CROP_VOXELS = 30_000_000   # 3× downsample above this


def generate_tissue_mesh(
    ct_crop: sitk.Image,
    mask_crop: sitk.Image,
    *,
    gauss_variance: float = 1.2,
    iso_value: float = 0.1,
    smooth_iterations: int = 30,
    smooth_passband: float = 0.08,
    max_faces: int = 500_000,
) -> vtk.vtkPolyData | None:
    """Smart tissue mesh extraction with auto-downsampling for large masks.

    For small crops (< 5 M voxels): full-resolution gradient-guided.
    For medium crops (5–30 M): 2× downsample before extraction.
    For huge crops (> 30 M): 3× downsample before extraction.

    Automatically computes adaptive decimation to stay under *max_faces*.

    Parameters
    ----------
    ct_crop, mask_crop : sitk.Image
        Co-registered CT and binary-mask crops (same size/spacing).
    gauss_variance : float
        Gaussian smoothing variance (mm²).
    iso_value : float
        FlyingEdges iso-surface threshold on gradient field.
    smooth_iterations, smooth_passband : int, float
        WindowedSinc smoothing parameters.
    max_faces : int
        Target upper bound on output face count.

    Returns
    -------
    vtk.vtkPolyData | None
        Post-processed mesh, or *None* if the result is empty.
    """
    crop_voxels = int(np.prod(mask_crop.GetSize()))

    # ── Auto-downsample large crops ──────────────────────────────────
    if crop_voxels > _HUGE_CROP_VOXELS:
        factor = 3
    elif crop_voxels > _LARGE_CROP_VOXELS:
        factor = 2
    else:
        factor = 1

    if factor > 1:
        ct_work = _downsample_sitk(sitk.Cast(ct_crop, sitk.sitkFloat32), factor)
        mask_work = _downsample_sitk(mask_crop, factor)
    else:
        ct_work = sitk.Cast(ct_crop, sitk.sitkFloat32)
        mask_work = mask_crop

    # ── Gradient-guided extraction ───────────────────────────────────
    smoothed = sitk.DiscreteGaussian(ct_work, variance=gauss_variance)
    gradient = sitk.GradientMagnitude(smoothed)
    mask_f = sitk.Cast(mask_work, sitk.sitkFloat32)
    refined = gradient * mask_f

    vtk_img = _sitk_to_vtk(refined, scalar_type="float")

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(vtk_img)
    fe.SetValue(0, iso_value)
    fe.ComputeNormalsOn()
    fe.Update()
    mesh = fe.GetOutput()

    if mesh.GetNumberOfCells() == 0:
        return None

    # ── Adaptive decimation ──────────────────────────────────────────
    raw_faces = mesh.GetNumberOfCells()
    if raw_faces > max_faces:
        target_reduction = 1.0 - (max_faces / raw_faces)
        target_reduction = min(target_reduction, 0.95)  # cap at 95%
    else:
        target_reduction = 0.5  # mild default

    mesh = improve_mesh(
        mesh,
        smooth_iterations=smooth_iterations,
        smooth_passband=smooth_passband,
        target_reduction=target_reduction,
    )

    return mesh


def merge_meshes(mesh_list: list[vtk.vtkPolyData]) -> vtk.vtkPolyData:
    """Merge multiple VTK meshes into a single unified mesh.

    Parameters
    ----------
    mesh_list : list[vtkPolyData]
        Meshes to combine (e.g. all vertebrae → spine).

    Returns
    -------
    vtk.vtkPolyData
        Single merged and cleaned mesh.
    """
    if len(mesh_list) == 1:
        return mesh_list[0]

    appender = vtk.vtkAppendPolyData()
    for m in mesh_list:
        appender.AddInputData(m)
    appender.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(appender.GetOutput())
    cleaner.Update()

    return cleaner.GetOutput()
