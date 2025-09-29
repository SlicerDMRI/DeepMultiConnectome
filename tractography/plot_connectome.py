'''
This script reads a connectome matrix from a CSV file and visualizes it as an image. 
The diagonal elements of the matrix can be set to zero before plotting. The output 
image is saved to a specified file.

Usage
python plot_connectome.py <csv_file> <output_file> <plot_name>

Example
python plot_connectome.py connectome_matrix.csv connectome_matrix.png "Connectome matrix for subject 120010"
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as mcolors

def plot_connectome(csv_file, output_file, name, zero_diagonal=False, log_scale=True, small_value=1e-6, difference=False):
    """
    Reads a connectome matrix from a CSV file and plots it as an image.

    Parameters:
    csv_file (str): Path to the CSV file containing the connectome matrix.
    output_file (str): Path to save the output image file.
    name (str): Title of the plot.
    zero_diagonal (bool): If True, sets the diagonal elements of the matrix to zero.
    log_scale (bool): If True, applies a logarithmic scale to the colormap.
    small_value (float): A small value added to the matrix to handle log scale for zero values.
    """
       # Read the CSV file directly into a NumPy array
    connectome_matrix = np.loadtxt(csv_file, delimiter=',')

    # Set the diagonal to zero
    if zero_diagonal:
        np.fill_diagonal(connectome_matrix, 0)
    
    if difference==True:
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=connectome_matrix.min(), vcenter=0, vmax=connectome_matrix.max())
    elif difference=='percent':
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    else:
        cmap = plt.get_cmap('jet')

        if log_scale:
            norm = mcolors.LogNorm(vmin=max(connectome_matrix.min(), 1), vmax=connectome_matrix.max())
            # Handle zero values by replacing them with a small positive value for log scale
            connectome_matrix = np.where(connectome_matrix == 0, small_value, connectome_matrix)
        else:
            norm = None

    # Plot the connectome matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(connectome_matrix, cmap=cmap, norm=norm)
    
    plt.colorbar(label='Connection Strength')
    plt.title(name)
    plt.xlabel('node')
    plt.ylabel('node')
    num_nodes=connectome_matrix.shape[0]
    plt.xticks(ticks=np.arange(9, num_nodes, 10), labels=np.arange(10, num_nodes, 10))
    plt.yticks(ticks=np.arange(9, num_nodes, 10), labels=np.arange(10, num_nodes, 10))
    
    # Save the plot as an image file
    plt.savefig(output_file, bbox_inches='tight', dpi=500)
    plt.close()
    # print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plot_connectome.py <csv_file> <output_file> <plot_name>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    name = sys.argv[3]
    plot_connectome(csv_file, output_file, name)
