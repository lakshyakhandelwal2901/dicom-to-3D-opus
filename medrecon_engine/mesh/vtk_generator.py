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
