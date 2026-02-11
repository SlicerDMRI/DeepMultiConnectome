#!/usr/bin/env python
"""
Convert TRK format tractography files to TCK format for use with MRtrix3
"""

import sys
import os
import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
import numpy as np

def convert_trk_to_tck(trk_path, tck_path):
    """Convert TRK file to TCK format"""
    print(f"Converting: {os.path.basename(trk_path)}")
    
    # Load TRK file
    trk = nib.streamlines.load(trk_path)
    
    # Get streamlines
    streamlines = trk.streamlines
    
    # Create TCK file with same header information
    # TCK format uses rasmm space
    tck_header = {
        'voxel_to_rasmm': trk.affine.copy(),
        'dimensions': trk.header['dimensions'],
        'voxel_sizes': trk.header['voxel_sizes'],
    }
    
    # Save as TCK
    nib.streamlines.save(
        nib.streamlines.Tractogram(streamlines, affine_to_rasmm=trk.affine),
        tck_path,
        header=tck_header
    )
    
    return len(streamlines)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_trk_to_tck.py <input.trk> <output.tck>")
        sys.exit(1)
    
    trk_path = sys.argv[1]
    tck_path = sys.argv[2]
    
    if not os.path.exists(trk_path):
        print(f"Error: Input file not found: {trk_path}")
        sys.exit(1)
    
    try:
        num_streamlines = convert_trk_to_tck(trk_path, tck_path)
        print(f"Success: Converted {num_streamlines} streamlines to {tck_path}")
    except Exception as e:
        print(f"Error converting file: {e}")
        sys.exit(1)
