import vtk

# Function to read a VTK file and convert it to VTP
def convert_vtk_to_vtp(vtk_filename, vtp_filename):
    # Create a reader for the VTK file
    vtk_reader = vtk.vtkPolyDataReader()
    vtk_reader.SetFileName(vtk_filename)
    vtk_reader.Update()
    
    # Get the PolyData from the reader
    poly_data = vtk_reader.GetOutput()
    
    # Create a writer for the VTP file
    vtp_writer = vtk.vtkXMLPolyDataWriter()
    vtp_writer.SetFileName(vtp_filename)
    vtp_writer.SetInputData(poly_data)
    
    # Write the VTP file
    vtp_writer.Write()
    print(f"Converted {vtk_filename} to {vtp_filename}")

# Example usage
vtk_filename = '/media/volume/connectomes_MV/TractCloud/TestData/HCP_MRTRIX/100307_fibers_10M.vtk'  # Replace with your VTK file
vtp_filename = '/media/volume/connectomes_MV/TractCloud/TestData/HCP_MRTRIX/100307_fibers_10M.vtp'  # Desired output VTP file name
convert_vtk_to_vtp(vtk_filename, vtp_filename)
vtk_filename = '/media/volume/connectomes_MV/TractCloud/TestData/HCP_MRTRIX/100408_fibers_10M.vtk'  # Replace with your VTK file
vtp_filename = '/media/volume/connectomes_MV/TractCloud/TestData/HCP_MRTRIX/100408_fibers_10M.vtp'  # Desired output VTP file name
convert_vtk_to_vtp(vtk_filename, vtp_filename)


a=pd_tractography

import vtk
import numpy as np

# Function to read a VTK file and print the number of points per line
def get_number_of_points_per_line(poly_data):

    # Get the number of lines (fibers)
    number_of_fibers = poly_data.GetNumberOfLines()
    print(f"Number of lines: {number_of_fibers}")

    # Initialize traversal of lines
    lines = poly_data.GetLines()
    lines.InitTraversal()
    
    line_ptids = vtk.vtkIdList()
    inpoints = poly_data.GetPoints()
    
    # Loop over lines and get the number of points for each line
    points_per_line = []
    if number_of_fibers<101:
        for lidx in range(number_of_fibers):
            lines.GetNextCell(line_ptids)
            line_length = line_ptids.GetNumberOfIds()
            points_per_line.append(line_length)

            print(f"Line {lidx}: Number of points = {line_length}")
    else:
        for lidx in range(100):
            lines.GetNextCell(line_ptids)
            line_length = line_ptids.GetNumberOfIds()
            points_per_line.append(line_length)

            print(f"Line {lidx}: Number of points = {line_length}")

    return points_per_line

# Example usage
points_per_line = get_number_of_points_per_line(pd_tractography)
print(points_per_line)

breakpoint() #GetPoints().GetPoint()
