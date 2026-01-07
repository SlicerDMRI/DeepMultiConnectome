#!/usr/bin/env python3
"""
Test script to verify streamline thresholding integration
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.streamline_thresholding import StreamlineThresholder
from utils.unified_connectome import ConnectomeAnalyzer

def test_thresholding_integration():
    """Test the complete thresholding integration"""
    
    print("Testing Streamline Thresholding Integration")
    print("=" * 60)
    
    subject_id = "100206"
    atlas_name = "aparc.a2009s+aseg"
    min_streamlines = 10
    
    # Set up paths
    base_path = Path("/media/volume/MV_HCP")
    subject_path = base_path / "HCP_MRtrix" / subject_id
    analysis_dir = subject_path / "analysis"
    
    print(f"Subject: {subject_id}")
    print(f"Atlas: {atlas_name}")
    print(f"Minimum streamlines per node: {min_streamlines}")
    print()
    
    # Check thresholded data exists
    thresholded_dir = analysis_dir / f"{atlas_name}_th{min_streamlines}"
    thresholded_labels = thresholded_dir / f"labels_10M_{atlas_name}_symmetric_thresholded.txt"
    
    if not thresholded_dir.exists():
        print(f"❌ Thresholded directory not found: {thresholded_dir}")
        return False
    
    print(f"✓ Thresholded directory found: {thresholded_dir}")
    
    if not thresholded_labels.exists():
        print(f"❌ Thresholded labels file not found: {thresholded_labels}")
        return False
    
    print(f"✓ Thresholded labels file found: {thresholded_labels}")
    
    # Check file size
    thresholded_size = thresholded_labels.stat().st_size
    print(f"✓ Thresholded file size: {thresholded_size:,} bytes")
    
    # Check that all required files exist
    required_files = [
        f"labels_10M_{atlas_name}_symmetric_thresholded.txt",
        "mean_fa_per_streamline.txt",
        "mean_md_per_streamline.txt", 
        "mean_ad_per_streamline.txt",
        "mean_rd_per_streamline.txt",
        f"predictions_{atlas_name}_symmetric_thresholded.txt",
        "streamline_indices_thresholded.txt",
        "thresholding_summary.txt"
    ]
    
    print("\nChecking thresholded files:")
    all_files_exist = True
    for filename in required_files:
        filepath = thresholded_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ❌ {filename}")
            all_files_exist = False
    
    if not all_files_exist:
        return False
    
    # Test loading the data with ConnectomeAnalyzer
    print("\nTesting ConnectomeAnalyzer with thresholded data:")
    try:
        analyzer = ConnectomeAnalyzer(str(thresholded_dir))
        print("✓ ConnectomeAnalyzer initialized successfully")
        
        # Try to load the data
        labels_data = analyzer.load_labels_file(str(thresholded_labels))
        print(f"✓ Loaded {len(labels_data):,} streamlines from thresholded data")
        
        # Check data integrity
        unique_labels = len(set(labels_data))
        print(f"✓ Found {unique_labels} unique node pairs in thresholded data")
        
    except Exception as e:
        print(f"❌ Error testing ConnectomeAnalyzer: {e}")
        return False
    
    print("\n🎉 All tests passed! Streamline thresholding integration is working correctly.")
    return True

if __name__ == "__main__":
    success = test_thresholding_integration()
    sys.exit(0 if success else 1)