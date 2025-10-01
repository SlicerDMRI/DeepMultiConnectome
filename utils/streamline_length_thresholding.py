#!/usr/bin/env python3
"""
Streamline Length-Based Thresholding Module

This module implements robust streamline-level thresholding based on streamline lengths.
It maintains data integrity by keeping all streamline properties synchronized.

Key features:
- Threshold based on m        # Save SIFT weights  
        sift_file = self.output_dir / "sift2_weights_thresholded.txt"
        np.savetxt(str(sift_file), thresholded_data['sift_weights'])mum and maximum streamline lengths
- Maintain synchronization between labels, diffusion metrics, and SIFT weights
- Store results in separate length-thresholded directories
- Preserve original data structure and pipeline compatibility
- Independent thresholding for true and predicted connectomes
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter, defaultdict
import shutil

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import create_logger


class StreamlineLengthThresholder:
    """
    Handles streamline-level thresholding based on streamline lengths
    """
    
    def __init__(self, subject_path: Union[str, Path], atlas: str, 
                 min_length: float = None, max_length: float = None, logger=None):
        """
        Initialize the length thresholder
        
        Args:
            subject_path: Path to subject directory
            atlas: Atlas name (e.g., 'aparc+aseg', 'aparc.a2009s+aseg')
            min_length: Minimum streamline length to keep (mm)
            max_length: Maximum streamline length to keep (mm)
            logger: Logger instance (optional)
        """
        self.subject_path = Path(subject_path)
        self.atlas = atlas
        self.min_length = min_length
        self.max_length = max_length
        self.logger = logger or create_logger(str(self.subject_path))
        
        # Setup paths
        self.base_output_dir = self.subject_path / "output"
        self.base_analysis_dir = self.subject_path / "analysis"
        
        # Create length threshold suffix
        length_suffix = f"len{int(min_length) if min_length else 'min'}-{int(max_length) if max_length else 'max'}"
        self.length_dir = f"{self.atlas}_{length_suffix}"
        
        # Output directories
        self.output_dir = self.base_analysis_dir / self.length_dir
        
        # Data file paths
        self.streamline_lengths_file = self.base_output_dir / "streamline_lengths_10M.txt"
        self.true_labels_file = self.base_output_dir / f"labels_10M_{self.atlas}_symmetric.txt"
        self.pred_labels_file = self.base_output_dir.parent / "TractCloud" / f"predictions_{self.atlas}_symmetric.txt"
        self.sift_weights_file = self.subject_path / "dMRI" / "sift2_weights.txt"
        
        # Diffusion metric files
        self.fa_file = self.subject_path / "dMRI" / "mean_fa_per_streamline.txt"
        self.md_file = self.subject_path / "dMRI" / "mean_md_per_streamline.txt"
        self.ad_file = self.subject_path / "dMRI" / "mean_ad_per_streamline.txt"
        self.rd_file = self.subject_path / "dMRI" / "mean_rd_per_streamline.txt"
        
        self.logger.info(f"Initialized length thresholder: {length_suffix}")
        
    def load_streamline_lengths(self) -> np.ndarray:
        """Load streamline lengths from file"""
        try:
            lengths = np.loadtxt(str(self.streamline_lengths_file))
            self.logger.info(f"Loaded {len(lengths)} streamline lengths")
            return lengths
        except Exception as e:
            self.logger.error(f"Failed to load streamline lengths: {e}")
            raise
            
    def load_data(self) -> Dict:
        """Load all necessary data for thresholding"""
        data = {}
        
        # Load streamline lengths
        data['lengths'] = self.load_streamline_lengths()
        
        # Load labels
        try:
            data['true_labels'] = np.loadtxt(str(self.true_labels_file), dtype=int)
            data['pred_labels'] = np.loadtxt(str(self.pred_labels_file), dtype=int)
            self.logger.info(f"Loaded {len(data['true_labels'])} true labels and {len(data['pred_labels'])} predicted labels")
        except Exception as e:
            self.logger.error(f"Failed to load labels: {e}")
            raise
            
        # Load diffusion metrics from separate files
        try:
            fa_values = np.loadtxt(str(self.fa_file))
            md_values = np.loadtxt(str(self.md_file))
            ad_values = np.loadtxt(str(self.ad_file))
            rd_values = np.loadtxt(str(self.rd_file))
            
            # Create a combined diffusion metrics dictionary
            data['diffusion'] = {
                'FA': fa_values,
                'MD': md_values,
                'AD': ad_values,
                'RD': rd_values
            }
            self.logger.info(f"Loaded diffusion metrics: FA({len(fa_values)}), MD({len(md_values)}), AD({len(ad_values)}), RD({len(rd_values)})")
        except Exception as e:
            self.logger.error(f"Failed to load diffusion metrics: {e}")
            raise
            
        # Load SIFT weights
        try:
            data['sift_weights'] = np.loadtxt(str(self.sift_weights_file))
            self.logger.info(f"Loaded {len(data['sift_weights'])} SIFT weights")
        except Exception as e:
            self.logger.error(f"Failed to load SIFT weights: {e}")
            raise
            
        # Verify consistency
        n_streamlines = len(data['lengths'])
        if not all(len(data[key]) == n_streamlines for key in ['true_labels', 'pred_labels', 'sift_weights']):
            raise ValueError("Inconsistent data lengths across files")
            
        # Check diffusion metrics consistency
        for metric_name, metric_values in data['diffusion'].items():
            if len(metric_values) != n_streamlines:
                raise ValueError(f"Diffusion metric {metric_name} length {len(metric_values)} doesn't match streamlines {n_streamlines}")
            
        self.logger.info(f"All data loaded successfully for {n_streamlines} streamlines")
        return data
        
    def apply_length_filter(self, lengths: np.ndarray) -> np.ndarray:
        """
        Apply length-based filtering to determine which streamlines to keep
        
        Args:
            lengths: Array of streamline lengths
            
        Returns:
            Boolean array indicating which streamlines to keep
        """
        keep_mask = np.ones(len(lengths), dtype=bool)
        
        # Apply minimum length filter
        if self.min_length is not None:
            min_mask = lengths >= self.min_length
            keep_mask &= min_mask
            removed_min = np.sum(~min_mask)
            self.logger.info(f"Min length filter ({self.min_length}mm): removed {removed_min} streamlines")
            
        # Apply maximum length filter
        if self.max_length is not None:
            max_mask = lengths <= self.max_length
            keep_mask &= max_mask
            removed_max = np.sum(~max_mask)
            self.logger.info(f"Max length filter ({self.max_length}mm): removed {removed_max} streamlines")
            
        total_kept = np.sum(keep_mask)
        total_removed = len(lengths) - total_kept
        retention_rate = (total_kept / len(lengths)) * 100
        
        self.logger.info(f"Length filtering: kept {total_kept}/{len(lengths)} streamlines ({retention_rate:.2f}%)")
        
        return keep_mask
        
    def apply_thresholding(self, data: Dict) -> Dict:
        """
        Apply length-based thresholding to all data
        
        Args:
            data: Dictionary containing all loaded data
            
        Returns:
            Dictionary containing thresholded data and statistics
        """
        self.logger.info("Starting length-based thresholding...")
        
        # Get streamlines to keep based on length criteria
        streamlines_to_keep = self.apply_length_filter(data['lengths'])
        
        # Apply filtering to all data
        thresholded_data = {
            'streamlines_to_keep': streamlines_to_keep,
            'original_count': len(data['lengths']),
            'thresholded_count': np.sum(streamlines_to_keep),
            'retention_rate': (np.sum(streamlines_to_keep) / len(data['lengths'])) * 100,
        }
        
        # Filter all data arrays
        thresholded_data['lengths'] = data['lengths'][streamlines_to_keep]
        thresholded_data['true_labels'] = data['true_labels'][streamlines_to_keep]
        thresholded_data['pred_labels'] = data['pred_labels'][streamlines_to_keep]
        thresholded_data['sift_weights'] = data['sift_weights'][streamlines_to_keep]
        
        # Filter diffusion metrics
        thresholded_data['diffusion'] = {}
        for metric_name, metric_values in data['diffusion'].items():
            thresholded_data['diffusion'][metric_name] = metric_values[streamlines_to_keep]
        
        # Calculate statistics for both connectomes
        true_stats = self._calculate_connectome_stats(thresholded_data['true_labels'], "True")
        pred_stats = self._calculate_connectome_stats(thresholded_data['pred_labels'], "Predicted")
        
        thresholded_data['true_stats'] = true_stats
        thresholded_data['pred_stats'] = pred_stats
        
        # Calculate length statistics
        length_stats = {
            'min_length': np.min(thresholded_data['lengths']),
            'max_length': np.max(thresholded_data['lengths']),
            'mean_length': np.mean(thresholded_data['lengths']),
            'median_length': np.median(thresholded_data['lengths']),
            'std_length': np.std(thresholded_data['lengths'])
        }
        thresholded_data['length_stats'] = length_stats
        
        self.logger.info(f"Thresholding complete: {thresholded_data['thresholded_count']}/{thresholded_data['original_count']} streamlines kept")
        
        return thresholded_data
        
    def _calculate_connectome_stats(self, labels: np.ndarray, connectome_type: str) -> Dict:
        """Calculate statistics for a connectome"""
        unique_nodes = np.unique(labels[labels != 0])  # Exclude background (0)
        node_counts = Counter(labels[labels != 0])
        
        stats = {
            'unique_nodes': len(unique_nodes),
            'total_connections': len(labels),
            'node_counts': dict(node_counts),
            'connectome_type': connectome_type
        }
        
        self.logger.info(f"{connectome_type} connectome: {len(unique_nodes)} unique nodes, {len(labels)} connections")
        return stats
        
    def save_thresholded_data(self, thresholded_data: Dict) -> Dict[str, str]:
        """Save thresholded data to files"""
        self.logger.info(f"Saving thresholded data to {self.output_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save streamline lengths
        lengths_file = self.output_dir / "streamline_lengths_thresholded.txt"
        np.savetxt(str(lengths_file), thresholded_data['lengths'])
        saved_files['lengths'] = str(lengths_file)
        
        # Save labels
        true_labels_file = self.output_dir / f"streamline_labels_{self.atlas}_true_thresholded.txt"
        np.savetxt(str(true_labels_file), thresholded_data['true_labels'], fmt='%d')
        saved_files['true_labels'] = str(true_labels_file)
        
        pred_labels_file = self.output_dir / f"streamline_labels_{self.atlas}_pred_thresholded.txt"
        np.savetxt(str(pred_labels_file), thresholded_data['pred_labels'], fmt='%d')
        saved_files['pred_labels'] = str(pred_labels_file)
        
        # Save diffusion metrics as separate files
        for metric_name, metric_values in thresholded_data['diffusion'].items():
            metric_file = self.output_dir / f"mean_{metric_name.lower()}_per_streamline_thresholded.txt"
            np.savetxt(str(metric_file), metric_values)
            saved_files[f'diffusion_{metric_name}'] = str(metric_file)
        
        # Save SIFT weights
        sift_file = self.output_dir / "sift_weights_thresholded.txt"
        np.savetxt(str(sift_file), thresholded_data['sift_weights'])
        saved_files['sift2_weights'] = str(sift_file)
        
        # Save summary report
        summary_file = self.output_dir / "length_thresholding_summary.txt"
        self._save_summary_report(thresholded_data, summary_file)
        saved_files['summary'] = str(summary_file)
        
        # Generate connectome CSV files for both true and predicted
        true_connectome_file = self.output_dir / f"{self.subject_path.name}_connectome_{self.atlas}_true.csv"
        pred_connectome_file = self.output_dir / f"{self.subject_path.name}_connectome_{self.atlas}_pred.csv"
        
        self._save_connectome_matrix(thresholded_data['true_labels'], true_connectome_file)
        self._save_connectome_matrix(thresholded_data['pred_labels'], pred_connectome_file)
        
        saved_files['true_connectome'] = str(true_connectome_file)
        saved_files['pred_connectome'] = str(pred_connectome_file)
        
        self.logger.info(f"Saved {len(saved_files)} files to {self.output_dir}")
        return saved_files
        
    def _save_connectome_matrix(self, labels: np.ndarray, output_file: Path):
        """Save connectome matrix as CSV"""
        # Create connectivity matrix
        unique_nodes = np.unique(labels[labels != 0])
        n_nodes = len(unique_nodes)
        
        # Create node mapping
        node_to_idx = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
        
        # Initialize connectivity matrix
        connectivity_matrix = np.zeros((n_nodes, n_nodes))
        
        # Count connections (assuming pairwise connections)
        for i in range(0, len(labels), 2):
            if i + 1 < len(labels):
                node1, node2 = labels[i], labels[i + 1]
                if node1 != 0 and node2 != 0 and node1 in node_to_idx and node2 in node_to_idx:
                    idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                    connectivity_matrix[idx1, idx2] += 1
                    connectivity_matrix[idx2, idx1] += 1  # Symmetric
        
        # Save as CSV with node labels
        df = pd.DataFrame(connectivity_matrix, 
                         index=sorted(unique_nodes), 
                         columns=sorted(unique_nodes))
        df.to_csv(str(output_file))
        
    def _save_summary_report(self, thresholded_data: Dict, output_file: Path):
        """Save detailed summary report"""
        with open(str(output_file), 'w') as f:
            f.write("Streamline Length-Based Thresholding Summary\n")
            f.write("=" * 45 + "\n")
            f.write(f"Atlas: {self.atlas}\n")
            if self.min_length:
                f.write(f"Minimum length: {self.min_length} mm\n")
            if self.max_length:
                f.write(f"Maximum length: {self.max_length} mm\n")
            f.write(f"Original streamline count: {thresholded_data['original_count']}\n")
            f.write(f"Thresholded streamline count: {thresholded_data['thresholded_count']}\n")
            f.write(f"Retention rate: {thresholded_data['retention_rate']:.2f}%\n\n")
            
            # Length statistics
            length_stats = thresholded_data['length_stats']
            f.write("=== LENGTH STATISTICS ===\n")
            f.write(f"Minimum length: {length_stats['min_length']:.2f} mm\n")
            f.write(f"Maximum length: {length_stats['max_length']:.2f} mm\n")
            f.write(f"Mean length: {length_stats['mean_length']:.2f} mm\n")
            f.write(f"Median length: {length_stats['median_length']:.2f} mm\n")
            f.write(f"Standard deviation: {length_stats['std_length']:.2f} mm\n\n")
            
            # True connectome statistics
            true_stats = thresholded_data['true_stats']
            f.write("=== TRUE CONNECTOME STATISTICS ===\n")
            f.write(f"Unique nodes: {true_stats['unique_nodes']}\n")
            f.write(f"Total connections: {true_stats['total_connections']}\n\n")
            
            # Predicted connectome statistics
            pred_stats = thresholded_data['pred_stats']
            f.write("=== PREDICTED CONNECTOME STATISTICS ===\n")
            f.write(f"Unique nodes: {pred_stats['unique_nodes']}\n")
            f.write(f"Total connections: {pred_stats['total_connections']}\n\n")
            
    def create_length_thresholded_dataset(self) -> Dict[str, str]:
        """
        Main function to create length-thresholded dataset
        
        Returns:
            Dictionary of saved file paths
        """
        try:
            self.logger.info(f"Starting length thresholding for subject {self.subject_path.name}")
            
            # Load data
            data = self.load_data()
            
            # Apply thresholding
            thresholded_data = self.apply_thresholding(data)
            
            # Save results
            saved_files = self.save_thresholded_data(thresholded_data)
            
            self.logger.info("Length thresholding completed successfully")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Length thresholding failed: {e}")
            raise


def threshold_subject_by_length(subject_path: Union[str, Path], atlas: str, 
                              min_length: float = None, max_length: float = None) -> Dict[str, str]:
    """
    Convenience function to threshold a subject's data by streamline length
    
    Args:
        subject_path: Path to subject directory
        atlas: Atlas name
        min_length: Minimum streamline length (mm)
        max_length: Maximum streamline length (mm)
        
    Returns:
        Dictionary of saved file paths
    """
    thresholder = StreamlineLengthThresholder(subject_path, atlas, min_length, max_length)
    return thresholder.create_length_thresholded_dataset()


def main():
    """Command-line interface for length thresholding"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Streamline Length-Based Thresholding')
    parser.add_argument('--subject-path', '-s', required=True, help='Subject directory path')
    parser.add_argument('--atlas', '-a', required=True, help='Atlas name')
    parser.add_argument('--min-length', '-min', type=float, default=None, 
                       help='Minimum streamline length in mm')
    parser.add_argument('--max-length', '-max', type=float, default=None,
                       help='Maximum streamline length in mm')
    
    args = parser.parse_args()
    
    if args.min_length is None and args.max_length is None:
        print("Error: At least one of --min-length or --max-length must be specified")
        return
    
    # Run length thresholding
    filter_desc = []
    if args.min_length:
        filter_desc.append(f"min {args.min_length}mm")
    if args.max_length:
        filter_desc.append(f"max {args.max_length}mm")
    
    print(f"Running length thresholding for {args.atlas} with {' and '.join(filter_desc)}...")
    
    saved_files = threshold_subject_by_length(
        subject_path=args.subject_path,
        atlas=args.atlas,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    if saved_files:
        print("\n✓ Length thresholding completed successfully!")
        print("Saved files:")
        for key, path in saved_files.items():
            print(f"  {key}: {path}")
    else:
        print("\n✗ Length thresholding failed!")


if __name__ == "__main__":
    main()