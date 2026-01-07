#!/usr/bin/env python3
"""
Test script for multi-subject connectome analysis

This script runs the analysis on a small subset of subjects to verify everything works
before running on the full dataset.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from multi_subject_connectome_analysis import MultiSubjectConnectomeAnalysis

def main():
    """Test with a small subset of subjects"""
    
    print("Testing Multi-Subject Connectome Analysis")
    print("="*50)
    
    # Use the subject list but limit to first 5 subjects for testing
    subject_list_file = "/media/volume/MV_HCP/subjects_tractography_output_1000_test.txt"
    output_dir = "/media/volume/HCP_diffusion_MV/DeepMultiConnectome/analysis/test_multi_subject"
    
    # Create test analysis with limited subjects
    analysis = MultiSubjectConnectomeAnalysis(
        subject_list_file=subject_list_file,
        output_dir=output_dir,
        max_subjects=5  # Test with only 5 subjects
    )
    
    # Run analysis with single process for easier debugging
    analysis.run_analysis(n_processes=1)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()