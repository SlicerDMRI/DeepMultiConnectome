#!/usr/bin/env python3
"""
Fix Data Mismatch Script

This script provides a solution to fix the data mismatch issue where diffusion metrics
have fewer values than expected due to NaN values being filtered out during parsing.

The solution ensures consistent array sizes by padding with NaN or using consistent indexing.
"""

import os
import numpy as np
from pathlib import Path

def analyze_nan_values_in_file(file_path: Path):
    """Analyze NaN values in a diffusion metric file"""
    
    print(f"\nAnalyzing NaN values in: {file_path.name}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the data line (skip comments)
    data_line = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            data_line = line
            break
    
    if not data_line:
        print("  No data line found")
        return
    
    values = data_line.split()
    total_values = len(values)
    
    nan_count = 0
    invalid_count = 0
    valid_count = 0
    
    nan_positions = []
    invalid_positions = []
    
    for i, val in enumerate(values):
        if val.lower() == 'nan':
            nan_count += 1
            nan_positions.append(i)
        else:
            try:
                float(val)
                valid_count += 1
            except ValueError:
                invalid_count += 1
                invalid_positions.append(i)
    
    print(f"  Total values in file: {total_values:,}")
    print(f"  Valid values: {valid_count:,}")
    print(f"  NaN values: {nan_count:,}")
    print(f"  Invalid values: {invalid_count:,}")
    print(f"  Missing values: {total_values - valid_count:,}")
    
    if nan_positions:
        print(f"  First 10 NaN positions: {nan_positions[:10]}")
    if invalid_positions:
        print(f"  First 10 invalid positions: {invalid_positions[:10]}")
    
    return {
        'total': total_values,
        'valid': valid_count,
        'nan': nan_count,
        'invalid': invalid_count,
        'nan_positions': nan_positions,
        'invalid_positions': invalid_positions
    }

def create_improved_parsing_function():
    """Create an improved parsing function that handles NaN values correctly"""
    
    function_code = '''
def load_diffusion_metric_improved(metric_file: Path, expected_length: int) -> np.ndarray:
    """
    Improved function to load diffusion metric values with proper NaN handling
    
    Args:
        metric_file: Path to the metric file
        expected_length: Expected number of values
        
    Returns:
        Array of metric values with NaN for missing/invalid values
    """
    with open(metric_file, 'r') as f:
        lines = f.readlines()
    
    # Find the data line (skip comments)
    data_line = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            data_line = line
            break
    
    if not data_line:
        # Return array of NaNs if no data found
        return np.full(expected_length, np.nan)
    
    values = data_line.split()
    
    # Create output array filled with NaN
    result = np.full(expected_length, np.nan)
    
    # Parse values, keeping NaN for invalid/missing values
    for i, val in enumerate(values):
        if i >= expected_length:
            break  # Don't exceed expected length
            
        if val.lower() == 'nan':
            result[i] = np.nan
        else:
            try:
                result[i] = float(val)
            except ValueError:
                result[i] = np.nan  # Keep as NaN for invalid values
    
    return result
'''
    
    return function_code

def test_improved_parsing():
    """Test the improved parsing function"""
    
    print("\n" + "="*80)
    print("TESTING IMPROVED PARSING FUNCTION")
    print("="*80)
    
    subject_id = "100206"
    base_path = Path("/media/volume/MV_HCP/HCP_MRtrix") / subject_id
    
    # Load reference length
    labels_file = base_path / "output" / "labels_10M_aparc+aseg_symmetric.txt"
    with open(labels_file, 'r') as f:
        expected_length = sum(1 for line in f if line.strip())
    
    print(f"Expected length: {expected_length:,}")
    
    # Test with FA file
    fa_file = base_path / "dMRI" / "mean_fa_per_streamline.txt"
    
    # Original parsing (from unified_connectome.py)
    print(f"\nOriginal parsing of {fa_file.name}:")
    with open(fa_file, 'r') as f:
        lines = f.readlines()
    
    metric_values_original = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            values = line.split()
            for val in values:
                if val.lower() != 'nan':  # Skip NaN values
                    try:
                        metric_values_original.append(float(val))
                    except ValueError:
                        continue
    
    print(f"  Original method result: {len(metric_values_original):,} values")
    
    # Improved parsing
    print(f"\nImproved parsing of {fa_file.name}:")
    
    # Execute the improved function (simplified version for testing)
    exec(create_improved_parsing_function(), globals())
    
    result_improved = load_diffusion_metric_improved(fa_file, expected_length)
    valid_count = np.sum(~np.isnan(result_improved))
    nan_count = np.sum(np.isnan(result_improved))
    
    print(f"  Improved method result: {len(result_improved):,} total values")
    print(f"  Valid values: {valid_count:,}")
    print(f"  NaN values: {nan_count:,}")
    print(f"  Array shape: {result_improved.shape}")

def create_fixed_unified_connectome_section():
    """Create the fixed section for unified_connectome.py"""
    
    fixed_code = '''
# FIXED VERSION for analyze_connectomes_from_labels function
# Replace the metric loading section with this improved version:

def load_metric_values_improved(metric_file: Path, expected_length: int, metric_name: str, logger=None) -> np.ndarray:
    """
    Improved metric loading that maintains consistent array sizes
    """
    try:
        with open(metric_file, 'r') as f:
            lines = f.readlines()
        
        # Find the data line (skip comments)
        data_line = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                data_line = line
                break
        
        if not data_line:
            if logger:
                logger.warning(f"No data found in {metric_name} file")
            return np.full(expected_length, np.nan)
        
        values = data_line.split()
        
        # Create output array
        result = np.full(expected_length, np.nan)
        
        valid_count = 0
        nan_count = 0
        invalid_count = 0
        
        # Parse values maintaining positions
        for i, val in enumerate(values):
            if i >= expected_length:
                break
                
            if val.lower() == 'nan':
                result[i] = np.nan
                nan_count += 1
            else:
                try:
                    result[i] = float(val)
                    valid_count += 1
                except ValueError:
                    result[i] = np.nan
                    invalid_count += 1
        
        # Log statistics
        if logger:
            total_in_file = len(values)
            if total_in_file != expected_length:
                logger.warning(f"Metric {metric_name}: file has {total_in_file} values, expected {expected_length}")
            
            logger.info(f"Metric {metric_name}: {valid_count} valid, {nan_count} NaN, {invalid_count} invalid values")
        
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading {metric_name} values: {e}")
        return np.full(expected_length, np.nan)

# Then in the main loop, replace:
#   metric_values = []
#   for line in lines:
#       ... (old parsing logic)
#
# With:
#   metric_values = load_metric_values_improved(metric_file, len(pred_labels), metric_name, logger)
#   
#   # Convert to list and apply filtering consistently
#   if length_filter_mask is not None:
#       metric_values = metric_values[length_filter_mask]
#   
#   metric_values = metric_values.tolist()
'''
    
    return fixed_code

def main():
    """Main analysis and solution function"""
    
    print("="*80)
    print("DATA MISMATCH SOLUTION ANALYSIS")
    print("="*80)
    
    subject_id = "100206"
    base_path = Path("/media/volume/MV_HCP/HCP_MRtrix") / subject_id
    
    # Analyze each diffusion metric file
    diffusion_files = {
        'fa': base_path / "dMRI" / "mean_fa_per_streamline.txt",
        'md': base_path / "dMRI" / "mean_md_per_streamline.txt",
        'ad': base_path / "dMRI" / "mean_ad_per_streamline.txt",
        'rd': base_path / "dMRI" / "mean_rd_per_streamline.txt",
        'sift2': base_path / "dMRI" / "sift2_weights.txt"
    }
    
    print("ANALYZING NaN AND INVALID VALUES:")
    
    for metric_name, file_path in diffusion_files.items():
        if file_path.exists():
            analyze_nan_values_in_file(file_path)
    
    # Test improved parsing
    test_improved_parsing()
    
    print("\n" + "="*80)
    print("SOLUTION SUMMARY")
    print("="*80)
    
    print("""
PROBLEM IDENTIFIED:
- Diffusion metric files contain NaN values that are filtered out during parsing
- This creates arrays of different sizes (2,868,497 instead of 2,878,459)
- When length filtering is applied, array indexing becomes inconsistent

SOLUTION:
1. Maintain consistent array sizes by preserving NaN positions
2. Use NumPy arrays with NaN for missing/invalid values
3. Apply the same filtering mask to all arrays simultaneously
4. Log discrepancies for transparency

IMPLEMENTATION:
The improved parsing function maintains array positions and logs statistics.
This ensures all arrays have the same size before filtering is applied.
""")
    
    print("\nFIXED CODE SECTION:")
    print(create_fixed_unified_connectome_section())
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Apply the fixed parsing logic to unified_connectome.py
2. Test with length filtering to ensure consistent array sizes
3. Verify that warning messages are resolved
4. Run complete analysis to confirm fix works
""")

if __name__ == "__main__":
    main()