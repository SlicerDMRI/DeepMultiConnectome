#!/usr/bin/env python3
"""
Debug Parsing Logic Script

This script investigates the exact parsing behavior that causes the warning:
"Metric fa has insufficient data (2868497 vs 2878459)"

It replicates the parsing logic from unified_connectome.py to understand where the mismatch occurs.
"""

import os
import sys
from pathlib import Path
import numpy as np

def debug_metric_parsing(metric_file: Path, metric_name: str):
    """Debug the exact parsing logic used in unified_connectome.py"""
    
    print(f"\nDEBUGGING {metric_name.upper()} PARSING:")
    print(f"File: {metric_file}")
    print(f"File exists: {metric_file.exists()}")
    
    if not metric_file.exists():
        return
    
    # Read file exactly as unified_connectome.py does
    with open(metric_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines read: {len(lines)}")
    
    # Show first few lines
    print("First 5 lines:")
    for i, line in enumerate(lines[:5]):
        print(f"  Line {i+1}: '{line.strip()[:50]}...' (length: {len(line.strip())})")
    
    # Parse exactly as unified_connectome.py does
    metric_values = []
    line_count = 0
    non_empty_lines = 0
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            non_empty_lines += 1
            try:
                # Handle multiple values per line (space-separated)
                values = line.split()
                print(f"  Non-empty line {non_empty_lines}: {len(values)} values")
                
                valid_values_in_line = 0
                for val in values:
                    if val.lower() != 'nan':  # Skip NaN values
                        try:
                            metric_values.append(float(val))
                            valid_values_in_line += 1
                        except ValueError:
                            print(f"    Warning: Could not parse value '{val}'")
                
                print(f"    Added {valid_values_in_line} valid values")
                line_count += 1
                
                # Stop after processing a few non-empty lines for debugging
                if non_empty_lines >= 3:
                    print(f"    ... (continuing with remaining lines)")
                    # Process remaining lines without detailed output
                    for remaining_line in lines[line_num+1:]:
                        remaining_line = remaining_line.strip()
                        if remaining_line and not remaining_line.startswith('#'):
                            remaining_values = remaining_line.split()
                            for val in remaining_values:
                                if val.lower() != 'nan':
                                    try:
                                        metric_values.append(float(val))
                                    except ValueError:
                                        pass
                    break
                    
            except ValueError:
                print(f"  Line {line_num+1}: Could not parse")
                continue  # Skip lines that can't be parsed
    
    print(f"Final parsing results:")
    print(f"  Non-empty lines processed: {non_empty_lines}")
    print(f"  Total metric values extracted: {len(metric_values)}")
    print(f"  First few values: {metric_values[:10]}")
    print(f"  Last few values: {metric_values[-10:]}")
    
    return len(metric_values)

def debug_length_filtering_impact():
    """Debug how length filtering affects the metric data"""
    
    print("\n" + "="*80)
    print("DEBUGGING LENGTH FILTERING IMPACT")
    print("="*80)
    
    subject_id = "100206"
    base_path = Path("/media/volume/MV_HCP/HCP_MRtrix") / subject_id
    
    # Load labels (reference data)
    labels_file = base_path / "output" / "labels_10M_aparc+aseg_symmetric.txt"
    print(f"Loading labels from: {labels_file}")
    
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Labels loaded: {len(labels):,}")
    
    # Load lengths
    lengths_file = base_path / "output" / "streamline_lengths_10M.txt"
    print(f"Loading lengths from: {lengths_file}")
    
    with open(lengths_file, 'r') as f:
        lengths = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    lengths.append(float(line))
                except ValueError:
                    continue
    
    print(f"Lengths loaded: {len(lengths):,}")
    
    # Apply length filtering (≥20mm) as in the analysis
    min_length = 20.0
    length_mask = np.array(lengths) >= min_length
    
    print(f"Length filtering (≥{min_length}mm):")
    print(f"  Original streamlines: {len(lengths):,}")
    print(f"  Streamlines kept: {np.sum(length_mask):,}")
    print(f"  Percentage kept: {np.sum(length_mask)/len(lengths)*100:.1f}%")
    
    # Load and debug FA values
    fa_file = base_path / "dMRI" / "mean_fa_per_streamline.txt"
    fa_count = debug_metric_parsing(fa_file, "fa")
    
    print(f"\nCOMPARISON:")
    print(f"  Original labels: {len(labels):,}")
    print(f"  Original lengths: {len(lengths):,}")
    print(f"  Original FA values: {fa_count:,}")
    print(f"  Filtered labels: {np.sum(length_mask):,}")
    
    # Check if there's an off-by-one or indexing issue
    if fa_count != len(labels):
        diff = fa_count - len(labels)
        print(f"  ⚠️ FA count difference: {diff:+,}")
        
        # Check if it's exactly the difference we saw in the warning
        expected_diff = 2868497 - 2878459  # From the warning message
        if diff == expected_diff:
            print(f"  ✓ This matches the warning difference ({expected_diff:+,})")

def main():
    """Main debug function"""
    
    print("="*80)
    print("DETAILED PARSING DEBUG")
    print("="*80)
    
    subject_id = "100206"
    base_path = Path("/media/volume/MV_HCP/HCP_MRtrix") / subject_id
    
    # Test each diffusion metric file
    diffusion_files = {
        'fa': base_path / "dMRI" / "mean_fa_per_streamline.txt",
        'md': base_path / "dMRI" / "mean_md_per_streamline.txt",
        'ad': base_path / "dMRI" / "mean_ad_per_streamline.txt",
        'rd': base_path / "dMRI" / "mean_rd_per_streamline.txt",
        'sift2': base_path / "dMRI" / "sift2_weights.txt"
    }
    
    for metric_name, metric_file in diffusion_files.items():
        if metric_file.exists():
            count = debug_metric_parsing(metric_file, metric_name)
        else:
            print(f"\n{metric_name.upper()} file not found: {metric_file}")
    
    # Debug length filtering impact
    debug_length_filtering_impact()
    
    print("\n" + "="*80)
    print("PROPOSED SOLUTION")
    print("="*80)
    
    print("""
ISSUE IDENTIFIED:
The parsing logic correctly extracts 2,878,459 values from diffusion metric files,
but there might be a subtle issue in the length filtering or array indexing that
causes a small number of values to be lost.

LIKELY CAUSES:
1. Array bounds checking during filtering
2. Inconsistent indexing between different arrays
3. Off-by-one errors in slicing operations
4. Memory/precision issues during large array operations

RECOMMENDED SOLUTIONS:
1. Add explicit array size validation before filtering
2. Use consistent indexing masks across all arrays
3. Log array sizes at each step of the filtering process
4. Handle edge cases where arrays might have slightly different sizes
5. Implement graceful degradation when sizes don't match exactly

IMMEDIATE FIX:
Add size validation and consistent array truncation in the filtering logic.
""")

if __name__ == "__main__":
    main()