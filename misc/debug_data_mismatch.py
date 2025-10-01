#!/usr/bin/env python3
"""
Debug Data Mismatch Script

This script analyzes the discrepancy between the number of streamlines 
and the diffusion metric data files to understand the source of warnings like:
"Metric fa has insufficient data (2868497 vs 2878459)"

It examines:
1. Streamline labels files
2. Diffusion metric files (FA, MD, AD, RD, SIFT2)
3. Streamline lengths files
4. File formats and data structure differences

Author: Analysis Assistant
Date: September 30, 2025
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd

def count_lines_in_file(file_path: Path) -> int:
    """Count total lines in a file"""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def count_values_in_file(file_path: Path, description: str = "") -> Tuple[int, int, List[str]]:
    """
    Count values in a file, handling different formats:
    - Line-based files (one value per line)
    - Space-separated files (multiple values per line)
    
    Returns:
        (total_values, total_lines, sample_lines)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        total_values = 0
        sample_lines = []
        
        for i, line in enumerate(lines[:10]):  # Sample first 10 lines
            line = line.strip()
            if line and not line.startswith('#'):
                # Count values in this line
                try:
                    values = line.split()
                    valid_values = 0
                    for val in values:
                        try:
                            float(val)
                            valid_values += 1
                        except ValueError:
                            if val.lower() != 'nan':  # Skip non-numeric values except nan
                                continue
                    
                    total_values += valid_values
                    sample_lines.append(f"Line {i+1}: '{line[:50]}...' -> {valid_values} values")
                    
                except Exception as e:
                    sample_lines.append(f"Line {i+1}: Error parsing - {e}")
        
        # If we only sampled, estimate total
        if len(lines) > 10:
            avg_values_per_line = total_values / min(10, len([l for l in lines[:10] if l.strip() and not l.startswith('#')]))
            valid_lines = len([l for l in lines if l.strip() and not l.startswith('#')])
            total_values = int(avg_values_per_line * valid_lines)
            sample_lines.append(f"Estimated total from sampling: {total_values} values")
        
        return total_values, total_lines, sample_lines
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return 0, 0, [f"Error: {e}"]

def analyze_file_structure(file_path: Path) -> Dict:
    """Analyze the structure of a data file"""
    
    info = {
        'file_path': str(file_path),
        'exists': file_path.exists(),
        'size_mb': 0,
        'total_lines': 0,
        'total_values': 0,
        'values_per_line_avg': 0,
        'sample_lines': [],
        'file_format': 'unknown'
    }
    
    if not file_path.exists():
        return info
    
    # Get file size
    info['size_mb'] = file_path.stat().st_size / (1024 * 1024)
    
    # Count lines and values
    total_values, total_lines, sample_lines = count_values_in_file(file_path)
    
    info['total_lines'] = total_lines
    info['total_values'] = total_values
    info['sample_lines'] = sample_lines
    
    if total_lines > 0:
        non_empty_lines = len([l for l in sample_lines if 'values' in l and not 'Error' in l])
        if non_empty_lines > 0:
            info['values_per_line_avg'] = total_values / max(total_lines - (total_lines - non_empty_lines), 1)
    
    # Determine file format
    if info['values_per_line_avg'] <= 1.1:  # Approximately 1 value per line
        info['file_format'] = 'line_based'
    elif info['values_per_line_avg'] > 1.1:
        info['file_format'] = 'space_separated'
    
    return info

def main():
    """Main analysis function"""
    
    print("="*80)
    print("DATA MISMATCH ANALYSIS")
    print("="*80)
    
    # Define subject and paths
    subject_id = "100206"
    base_path = Path("/media/volume/MV_HCP/HCP_MRtrix") / subject_id
    
    # Define file paths
    files_to_analyze = {
        'streamline_labels_aparc+aseg': base_path / "output" / "labels_10M_aparc+aseg_symmetric.txt",
        'streamline_labels_aparc.a2009s+aseg': base_path / "output" / "labels_10M_aparc.a2009s+aseg_symmetric.txt",
        'predicted_labels_aparc+aseg': base_path / "TractCloud" / "predictions_aparc+aseg_symmetric.txt",
        'predicted_labels_aparc.a2009s+aseg': base_path / "TractCloud" / "predictions_aparc.a2009s+aseg_symmetric.txt",
        'streamline_lengths': base_path / "output" / "streamline_lengths_10M.txt",
        'fa_values': base_path / "dMRI" / "mean_fa_per_streamline.txt",
        'md_values': base_path / "dMRI" / "mean_md_per_streamline.txt", 
        'ad_values': base_path / "dMRI" / "mean_ad_per_streamline.txt",
        'rd_values': base_path / "dMRI" / "mean_rd_per_streamline.txt",
        'sift2_weights': base_path / "dMRI" / "sift2_weights.txt"
    }
    
    print(f"Analyzing files for subject {subject_id}")
    print(f"Base path: {base_path}")
    print()
    
    # Analyze each file
    results = {}
    for name, file_path in files_to_analyze.items():
        print(f"Analyzing {name}...")
        results[name] = analyze_file_structure(file_path)
        
        # Print summary
        info = results[name]
        if info['exists']:
            print(f"  ✓ {info['size_mb']:.1f} MB, {info['total_lines']:,} lines, {info['total_values']:,} values")
            print(f"    Format: {info['file_format']}, Avg values/line: {info['values_per_line_avg']:.2f}")
        else:
            print(f"  ✗ File not found")
        print()
    
    # Create summary comparison
    print("="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    # Group by type
    streamline_files = {k: v for k, v in results.items() if 'labels' in k or 'lengths' in k}
    diffusion_files = {k: v for k, v in results.items() if any(x in k for x in ['fa', 'md', 'ad', 'rd', 'sift2'])}
    
    print("STREAMLINE-RELATED FILES:")
    print(f"{'File':<35} {'Lines':<10} {'Values':<10} {'Format':<15}")
    print("-" * 70)
    for name, info in streamline_files.items():
        if info['exists']:
            print(f"{name:<35} {info['total_lines']:<10,} {info['total_values']:<10,} {info['file_format']:<15}")
    
    print("\nDIFFUSION METRIC FILES:")
    print(f"{'File':<35} {'Lines':<10} {'Values':<10} {'Format':<15}")
    print("-" * 70)
    for name, info in diffusion_files.items():
        if info['exists']:
            print(f"{name:<35} {info['total_lines']:<10,} {info['total_values']:<10,} {info['file_format']:<15}")
    
    # Identify discrepancies
    print("\n" + "="*80)
    print("DISCREPANCY ANALYSIS")
    print("="*80)
    
    # Get reference counts
    ref_labels = None
    ref_lengths = None
    
    for name, info in streamline_files.items():
        if info['exists'] and 'aparc+aseg' in name and 'labels' in name:
            ref_labels = info['total_values']
            print(f"Reference streamline count (from {name}): {ref_labels:,}")
            break
    
    for name, info in streamline_files.items():
        if info['exists'] and 'lengths' in name:
            ref_lengths = info['total_values']
            print(f"Reference length count (from {name}): {ref_lengths:,}")
            break
    
    if ref_labels:
        print(f"\nComparing diffusion metrics to reference count ({ref_labels:,}):")
        for name, info in diffusion_files.items():
            if info['exists']:
                diff = info['total_values'] - ref_labels
                percent_diff = (diff / ref_labels) * 100
                status = "✓" if abs(diff) < 100 else "⚠" if abs(diff) < 10000 else "✗"
                print(f"  {status} {name:<25}: {info['total_values']:>10,} ({diff:+,}, {percent_diff:+.2f}%)")
    
    # Analyze potential causes
    print("\n" + "="*80)
    print("POTENTIAL CAUSES & SOLUTIONS")
    print("="*80)
    
    print("1. FILE FORMAT DIFFERENCES:")
    line_based = [name for name, info in results.items() if info.get('file_format') == 'line_based']
    space_separated = [name for name, info in results.items() if info.get('file_format') == 'space_separated']
    
    print(f"   Line-based files (1 value per line): {line_based}")
    print(f"   Space-separated files (multiple values per line): {space_separated}")
    
    print("\n2. POSSIBLE EXPLANATIONS:")
    if ref_labels and any(info['exists'] and abs(info['total_values'] - ref_labels) > 1000 for info in diffusion_files.values()):
        print("   - Diffusion metrics may have been computed on a subset of streamlines")
        print("   - Some streamlines might have failed diffusion metric computation")
        print("   - Files may have been generated with different parameters or versions")
        print("   - Quality control filters may have been applied to diffusion data")
    
    print("\n3. RECOMMENDED SOLUTIONS:")
    print("   a) Use consistent array indexing when filtering by length")
    print("   b) Apply the same filtering mask to all data arrays simultaneously")
    print("   c) Handle missing values gracefully (pad with NaN or truncate consistently)")
    print("   d) Log discrepancies and document which approach is used")
    
    # Generate detailed file samples
    print("\n" + "="*80)
    print("FILE SAMPLE ANALYSIS")
    print("="*80)
    
    for name, info in results.items():
        if info['exists'] and info['sample_lines']:
            print(f"\n{name}:")
            for sample_line in info['sample_lines'][:5]:  # Show first 5 samples
                print(f"  {sample_line}")
    
    # Create a summary CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Convert results to DataFrame
    df_data = []
    for name, info in results.items():
        df_data.append({
            'File_Name': name,
            'File_Path': info['file_path'],
            'Exists': info['exists'],
            'Size_MB': info['size_mb'],
            'Total_Lines': info['total_lines'],
            'Total_Values': info['total_values'],
            'Values_Per_Line': info['values_per_line_avg'],
            'File_Format': info['file_format']
        })
    
    df = pd.DataFrame(df_data)
    output_file = Path("data_mismatch_analysis.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(results, ref_labels)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def create_visualization(results: Dict, ref_count: Optional[int]):
    """Create visualization of the data mismatch analysis"""
    
    # Prepare data for plotting
    names = []
    values = []
    colors = []
    
    for name, info in results.items():
        if info['exists']:
            names.append(name.replace('_', '\n'))
            values.append(info['total_values'])
            
            # Color coding
            if 'labels' in name or 'lengths' in name:
                colors.append('blue')
            elif any(metric in name for metric in ['fa', 'md', 'ad', 'rd']):
                colors.append('red')
            elif 'sift2' in name:
                colors.append('green')
            else:
                colors.append('gray')
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Bar chart of value counts
    bars = ax1.bar(range(len(names)), values, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Number of Values')
    ax1.set_title('Data Count Comparison Across Files')
    ax1.grid(True, alpha=0.3)
    
    # Add reference line if available
    if ref_count:
        ax1.axhline(y=ref_count, color='black', linestyle='--', alpha=0.8, 
                   label=f'Reference count: {ref_count:,}')
        ax1.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Difference from reference (if available)
    if ref_count:
        differences = [v - ref_count for v in values]
        colors_diff = ['green' if abs(d) < 1000 else 'orange' if abs(d) < 10000 else 'red' 
                      for d in differences]
        
        bars2 = ax2.bar(range(len(names)), differences, color=colors_diff, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Difference from Reference')
        ax2.set_title(f'Difference from Reference Count ({ref_count:,})')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars2, differences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (max(differences) - min(differences))*0.01 if height >= 0 else height - (max(differences) - min(differences))*0.01,
                    f'{diff:+,}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('data_mismatch_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualization saved to: data_mismatch_visualization.png")

if __name__ == "__main__":
    main()