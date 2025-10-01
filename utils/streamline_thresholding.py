#!/usr/bin/env python3
"""
Streamline Thresholding Module

This module implements robust streamline-level thresholding based on node connectivity.
It maintains data integrity by keeping all streamline properties synchronized.

Key features:
- Threshold based on number of streamlines per node
- Maintain synchronization between labels, diffusion metrics, and SIFT weights
- Store results in separate thresholded directories
- Preserve original data structure and pipeline compatibility
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


class StreamlineThresholder:
    """
    Handles streamline-level thresholding based on node connectivity
    """
    
    def __init__(self, subject_path: Union[str, Path], atlas: str, 
                 min_streamlines_per_node: int = 10, logger=None):
        """
        Initialize the thresholder
        
        Args:
            subject_path: Path to subject directory
            atlas: Atlas name (e.g., 'aparc+aseg', 'aparc.a2009s+aseg')
            min_streamlines_per_node: Minimum streamlines required per node to keep
            logger: Logger instance (optional)
        """
        self.subject_path = Path(subject_path)
        self.atlas = atlas
        self.min_streamlines_per_node = min_streamlines_per_node
        self.logger = logger or create_logger(str(self.subject_path))
        
        # Setup paths
        self.diffusion_dir = self.subject_path / "dMRI"
        self.output_dir = self.subject_path / "output"
        self.tractcloud_dir = self.subject_path / "TractCloud"
        
        # Create thresholded output directory
        self.threshold_suffix = f"_th{min_streamlines_per_node}"
        self.thresholded_dir = self.subject_path / "analysis" / f"{atlas}{self.threshold_suffix}"
        self.thresholded_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized StreamlineThresholder for {atlas} with threshold {min_streamlines_per_node}")
        self.logger.info(f"Thresholded results will be saved to: {self.thresholded_dir}")
    
    def load_streamline_data(self) -> Dict:
        """
        Load all streamline data into a synchronized structure
        
        Returns:
            Dictionary with all streamline data
        """
        self.logger.info("Loading streamline data...")
        
        data = {
            'true_labels': None,
            'pred_labels': None,
            'diffusion_metrics': {},
            'sift_weights': None,
            'streamline_indices': None
        }
        
        # Load true labels
        true_labels_file = self.output_dir / f"labels_10M_{self.atlas}_symmetric.txt"
        if true_labels_file.exists():
            with open(true_labels_file, 'r') as f:
                data['true_labels'] = [int(line.strip()) for line in f if line.strip()]
            self.logger.info(f"Loaded {len(data['true_labels'])} true labels")
        else:
            self.logger.warning(f"True labels file not found: {true_labels_file}")
        
        # Load predicted labels
        pred_labels_file = self.tractcloud_dir / f"predictions_{self.atlas}_symmetric.txt"
        if pred_labels_file.exists():
            with open(pred_labels_file, 'r') as f:
                data['pred_labels'] = [int(line.strip()) for line in f if line.strip()]
            self.logger.info(f"Loaded {len(data['pred_labels'])} predicted labels")
        else:
            self.logger.warning(f"Predicted labels file not found: {pred_labels_file}")
        
        # Load diffusion metrics
        for metric in ['fa', 'md', 'ad', 'rd']:
            metric_file = self.diffusion_dir / f"mean_{metric}_per_streamline.txt"
            if metric_file.exists():
                metric_values = self._load_metric_file(metric_file)
                if metric_values:
                    data['diffusion_metrics'][metric] = metric_values
                    self.logger.info(f"Loaded {len(metric_values)} {metric.upper()} values")
            else:
                self.logger.warning(f"Metric file not found: {metric_file}")
        
        # Load SIFT2 weights
        sift_file = self.diffusion_dir / "sift2_weights.txt"
        if sift_file.exists():
            sift_values = self._load_sift2_file(sift_file)
            if sift_values:
                data['sift_weights'] = sift_values
                self.logger.info(f"Loaded {len(sift_values)} SIFT2 weights")
        else:
            self.logger.warning(f"SIFT2 weights file not found: {sift_file}")
        
        # Create streamline indices for tracking
        if data['true_labels']:
            data['streamline_indices'] = list(range(len(data['true_labels'])))
        
        return data
    
    def _load_sift2_file(self, file_path: Path) -> List[float]:
        """
        Load SIFT2 weights from file (weights only, no count header)
        
        Args:
            file_path: Path to SIFT2 weights file
            
        Returns:
            List of SIFT2 weights
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # SIFT2 files contain only weights - no header with count
            # Parse all lines as space-separated floating point values
            weight_data = []
            for line in lines:
                line = line.strip()
                if line:
                    # Split line by whitespace and add all valid weights
                    for weight_str in line.split():
                        try:
                            weight = float(weight_str)
                            weight_data.append(weight)
                        except ValueError:
                            self.logger.warning(f"Skipping invalid SIFT2 weight: {weight_str}")
            
            self.logger.info(f"Loaded {len(weight_data)} SIFT2 weights from {file_path}")
            return weight_data
            
        except Exception as e:
            self.logger.error(f"Error loading SIFT2 file {file_path}: {e}")
            return []

    def _load_metric_file(self, file_path: Path) -> List[float]:
        """
        Load metric values from file, handling various formats
        
        Args:
            file_path: Path to metric file
            
        Returns:
            List of metric values
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the data line (skip comments) - FA/MD/AD/RD files have all values on one line
            data_line = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    data_line = line
                    break
            
            if not data_line:
                self.logger.warning(f"No data found in {file_path}")
                return []
            
            # Split the line to get all values
            values = []
            value_strings = data_line.split()
            
            for val_str in value_strings:
                if val_str.lower() not in ['nan', 'inf', '-inf']:
                    try:
                        values.append(float(val_str))
                    except ValueError:
                        values.append(np.nan)  # Keep NaN for consistency
                else:
                    values.append(np.nan)  # Keep NaN for consistency
            
            return values
            
        except Exception as e:
            self.logger.error(f"Error loading metric file {file_path}: {e}")
            return []
    
    def analyze_node_connectivity(self, labels: List[int]) -> Dict:
        """
        Analyze node connectivity to determine which nodes to keep
        
        Args:
            labels: List of node labels for each streamline
            
        Returns:
            Dictionary with connectivity analysis
        """
        self.logger.info("Analyzing node connectivity...")
        
        # Count streamlines per node
        node_counts = Counter(labels)
        
        # Determine nodes to keep
        nodes_to_keep = set()
        nodes_to_remove = set()
        
        for node, count in node_counts.items():
            if count >= self.min_streamlines_per_node:
                nodes_to_keep.add(node)
            else:
                nodes_to_remove.add(node)
        
        analysis = {
            'total_nodes': len(node_counts),
            'nodes_to_keep': nodes_to_keep,
            'nodes_to_remove': nodes_to_remove,
            'nodes_kept_count': len(nodes_to_keep),
            'nodes_removed_count': len(nodes_to_remove),
            'node_counts': node_counts,
            'streamlines_kept': sum(count for node, count in node_counts.items() 
                                  if node in nodes_to_keep),
            'streamlines_removed': sum(count for node, count in node_counts.items() 
                                     if node in nodes_to_remove)
        }
        
        self.logger.info(f"Node analysis complete:")
        self.logger.info(f"  Total nodes: {analysis['total_nodes']}")
        self.logger.info(f"  Nodes to keep: {analysis['nodes_kept_count']}")
        self.logger.info(f"  Nodes to remove: {analysis['nodes_removed_count']}")
        self.logger.info(f"  Streamlines to keep: {analysis['streamlines_kept']}")
        self.logger.info(f"  Streamlines to remove: {analysis['streamlines_removed']}")
        
        return analysis
    
    def apply_thresholding(self, data: Dict) -> Dict:
        """
        Apply thresholding to all data with independent thresholding for true and predicted connectomes
        
        Args:
            data: Dictionary with all streamline data
            
        Returns:
            Dictionary with thresholded data for both true and predicted connectomes
        """
        self.logger.info(f"Applying independent thresholding with minimum {self.min_streamlines_per_node} streamlines per node...")
        
        if not data['true_labels']:
            self.logger.error("No true labels found - cannot apply thresholding")
            return {}
        
        # Analyze connectivity for both true and predicted labels
        true_analysis = self.analyze_node_connectivity(data['true_labels'])
        
        pred_analysis = None
        if data['pred_labels']:
            pred_analysis = self.analyze_node_connectivity(data['pred_labels'])
        
        # Determine which streamlines to keep for TRUE connectome
        true_streamlines_to_keep = []
        true_nodes_to_keep = true_analysis['nodes_to_keep']
        
        for i, label in enumerate(data['true_labels']):
            if label in true_nodes_to_keep:
                true_streamlines_to_keep.append(i)
        
        self.logger.info(f"TRUE connectome: Keeping {len(true_streamlines_to_keep)} out of {len(data['true_labels'])} streamlines")
        
        # Determine which streamlines to keep for PREDICTED connectome (independent thresholding)
        pred_streamlines_to_keep = []
        if data['pred_labels'] and pred_analysis:
            pred_nodes_to_keep = pred_analysis['nodes_to_keep']
            
            for i, label in enumerate(data['pred_labels']):
                if label in pred_nodes_to_keep:
                    pred_streamlines_to_keep.append(i)
            
            self.logger.info(f"PREDICTED connectome: Keeping {len(pred_streamlines_to_keep)} out of {len(data['pred_labels'])} streamlines")
        
        # Apply filtering to all data
        thresholded_data = {
            'true_streamlines_to_keep': true_streamlines_to_keep,
            'pred_streamlines_to_keep': pred_streamlines_to_keep,
            'true_analysis': true_analysis,
            'pred_analysis': pred_analysis,
            'original_count': len(data['true_labels']),
            'true_thresholded_count': len(true_streamlines_to_keep),
            'pred_thresholded_count': len(pred_streamlines_to_keep) if pred_streamlines_to_keep else 0
        }
        
        # Filter true labels using true thresholding
        if data['true_labels']:
            thresholded_data['true_labels'] = [data['true_labels'][i] for i in true_streamlines_to_keep]
        
        # Filter predicted labels using predicted thresholding  
        if data['pred_labels'] and pred_streamlines_to_keep:
            # Ensure indices are within bounds
            min_len = len(data['pred_labels'])
            filtered_indices = [i for i in pred_streamlines_to_keep if i < min_len]
            thresholded_data['pred_labels'] = [data['pred_labels'][i] for i in filtered_indices]
        
        # For diffusion metrics and SIFT weights, we'll use true thresholding as reference
        # since these are properties of the actual streamlines (ground truth based)
        
        # Filter diffusion metrics using true thresholding
        thresholded_data['diffusion_metrics'] = {}
        for metric, values in data['diffusion_metrics'].items():
            if values:
                min_len = min(len(data['true_labels']), len(values))
                filtered_indices = [i for i in true_streamlines_to_keep if i < min_len]
                thresholded_data['diffusion_metrics'][metric] = [values[i] for i in filtered_indices]
        
        # Filter SIFT weights using true thresholding
        if data['sift_weights']:
            min_len = min(len(data['true_labels']), len(data['sift_weights']))
            filtered_indices = [i for i in true_streamlines_to_keep if i < min_len]
            thresholded_data['sift_weights'] = [data['sift_weights'][i] for i in filtered_indices]
        
        # Store both sets of streamline indices for reference
        thresholded_data['true_streamline_indices'] = true_streamlines_to_keep
        if pred_streamlines_to_keep:
            thresholded_data['pred_streamline_indices'] = pred_streamlines_to_keep
        
        return thresholded_data
    
    def save_thresholded_data(self, thresholded_data: Dict) -> Dict[str, str]:
        """
        Save thresholded data to files in the thresholded directory
        
        Args:
            thresholded_data: Dictionary with thresholded data
            
        Returns:
            Dictionary with paths to saved files
        """
        self.logger.info(f"Saving thresholded data to {self.thresholded_dir}")
        
        saved_files = {}
        
        # Save true labels (thresholded based on true connectivity)
        if 'true_labels' in thresholded_data:
            true_labels_file = self.thresholded_dir / f"labels_10M_{self.atlas}_symmetric_thresholded.txt"
            with open(true_labels_file, 'w') as f:
                for label in thresholded_data['true_labels']:
                    f.write(f"{label}\n")
            saved_files['true_labels'] = str(true_labels_file)
            self.logger.info(f"Saved {len(thresholded_data['true_labels'])} true labels to {true_labels_file}")
        
        # Save predicted labels (thresholded based on predicted connectivity)
        if 'pred_labels' in thresholded_data:
            pred_labels_file = self.thresholded_dir / f"predictions_{self.atlas}_symmetric_thresholded.txt"
            with open(pred_labels_file, 'w') as f:
                for label in thresholded_data['pred_labels']:
                    f.write(f"{label}\n")
            saved_files['pred_labels'] = str(pred_labels_file)
            self.logger.info(f"Saved {len(thresholded_data['pred_labels'])} predicted labels to {pred_labels_file}")
        
        # Save diffusion metrics (using true thresholding as these are ground truth properties)
        for metric, values in thresholded_data.get('diffusion_metrics', {}).items():
            metric_file = self.thresholded_dir / f"mean_{metric}_per_streamline.txt"
            with open(metric_file, 'w') as f:
                # Add a header comment line
                f.write(f"# {metric.upper()} values per streamline (thresholded based on true connectivity)\n")
                # Write all values on a single line (space-separated) to match original format
                f.write(' '.join(str(value) for value in values) + '\n')
            saved_files[f'{metric}_metrics'] = str(metric_file)
            self.logger.info(f"Saved {len(values)} {metric.upper()} values to {metric_file}")
        
        # Save SIFT2 weights (using true thresholding as these are ground truth properties)
        if 'sift_weights' in thresholded_data:
            sift_file = self.thresholded_dir / "sift2_weights.txt"
            with open(sift_file, 'w') as f:
                # SIFT2 weights format: all values space-separated (no count header)
                f.write(' '.join(str(weight) for weight in thresholded_data['sift_weights']) + '\n')
            saved_files['sift_weights'] = str(sift_file)
            self.logger.info(f"Saved {len(thresholded_data['sift_weights'])} SIFT2 weights to {sift_file}")
        
        # Save streamline indices for both true and predicted
        if 'true_streamline_indices' in thresholded_data:
            true_indices_file = self.thresholded_dir / "true_streamline_indices_thresholded.txt"
            with open(true_indices_file, 'w') as f:
                for idx in thresholded_data['true_streamline_indices']:
                    f.write(f"{idx}\n")
            saved_files['true_streamline_indices'] = str(true_indices_file)
        
        if 'pred_streamline_indices' in thresholded_data:
            pred_indices_file = self.thresholded_dir / "pred_streamline_indices_thresholded.txt"
            with open(pred_indices_file, 'w') as f:
                for idx in thresholded_data['pred_streamline_indices']:
                    f.write(f"{idx}\n")
            saved_files['pred_streamline_indices'] = str(pred_indices_file)
        
        # Save thresholding summary with details for both connectomes
        summary_file = self.thresholded_dir / "thresholding_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Independent Streamline Thresholding Summary\n")
            f.write(f"==========================================\n")
            f.write(f"Atlas: {self.atlas}\n")
            f.write(f"Minimum streamlines per node: {self.min_streamlines_per_node}\n")
            f.write(f"Original streamline count: {thresholded_data.get('original_count', 'N/A')}\n")
            f.write(f"\n=== TRUE CONNECTOME THRESHOLDING ===\n")
            f.write(f"Thresholded streamline count: {thresholded_data.get('true_thresholded_count', 'N/A')}\n")
            f.write(f"Retention rate: {thresholded_data.get('true_thresholded_count', 0) / thresholded_data.get('original_count', 1) * 100:.2f}%\n")
            
            if 'true_analysis' in thresholded_data:
                analysis = thresholded_data['true_analysis']
                f.write(f"\nTrue Connectome Node Analysis:\n")
                f.write(f"  Total nodes: {analysis['total_nodes']}\n")
                f.write(f"  Nodes kept: {analysis['nodes_kept_count']}\n")
                f.write(f"  Nodes removed: {analysis['nodes_removed_count']}\n")
                f.write(f"  Streamlines kept: {analysis['streamlines_kept']}\n")
                f.write(f"  Streamlines removed: {analysis['streamlines_removed']}\n")
                f.write(f"\nTrue Connectome - Nodes removed (with streamline counts):\n")
                for node in sorted(analysis['nodes_to_remove']):
                    count = analysis['node_counts'].get(node, 0)
                    f.write(f"  Node {node}: {count} streamlines\n")
                f.write(f"\nTrue Connectome - Nodes kept summary:\n")
                kept_counts = [analysis['node_counts'][node] for node in analysis['nodes_to_keep']]
                if kept_counts:
                    f.write(f"  Min streamlines in kept nodes: {min(kept_counts)}\n")
                    f.write(f"  Max streamlines in kept nodes: {max(kept_counts)}\n")
                    f.write(f"  Mean streamlines in kept nodes: {np.mean(kept_counts):.1f}\n")
            
            # Add predicted connectome analysis if available
            if 'pred_analysis' in thresholded_data and thresholded_data['pred_analysis']:
                f.write(f"\n=== PREDICTED CONNECTOME THRESHOLDING ===\n")
                f.write(f"Thresholded streamline count: {thresholded_data.get('pred_thresholded_count', 'N/A')}\n")
                f.write(f"Retention rate: {thresholded_data.get('pred_thresholded_count', 0) / thresholded_data.get('original_count', 1) * 100:.2f}%\n")
                
                pred_analysis = thresholded_data['pred_analysis']
                f.write(f"\nPredicted Connectome Node Analysis:\n")
                f.write(f"  Total nodes: {pred_analysis['total_nodes']}\n")
                f.write(f"  Nodes kept: {pred_analysis['nodes_kept_count']}\n")
                f.write(f"  Nodes removed: {pred_analysis['nodes_removed_count']}\n")
                f.write(f"  Streamlines kept: {pred_analysis['streamlines_kept']}\n")
                f.write(f"  Streamlines removed: {pred_analysis['streamlines_removed']}\n")
                f.write(f"\nPredicted Connectome - Nodes removed (with streamline counts):\n")
                for node in sorted(pred_analysis['nodes_to_remove']):
                    count = pred_analysis['node_counts'].get(node, 0)
                    f.write(f"  Node {node}: {count} streamlines\n")
                f.write(f"\nPredicted Connectome - Nodes kept summary:\n")
                pred_kept_counts = [pred_analysis['node_counts'][node] for node in pred_analysis['nodes_to_keep']]
                if pred_kept_counts:
                    f.write(f"  Min streamlines in kept nodes: {min(pred_kept_counts)}\n")
                    f.write(f"  Max streamlines in kept nodes: {max(pred_kept_counts)}\n")
                    f.write(f"  Mean streamlines in kept nodes: {np.mean(pred_kept_counts):.1f}\n")
            
            f.write(f"\nSaved files:\n")
            for key, path in saved_files.items():
                f.write(f"  {key}: {path}\n")
        
        saved_files['summary'] = str(summary_file)
        self.logger.info(f"Saved thresholding summary to {summary_file}")
        
        return saved_files
    
    def create_thresholded_dataset(self) -> Dict[str, str]:
        """
        Complete pipeline to create thresholded dataset
        
        Returns:
            Dictionary with paths to thresholded files
        """
        self.logger.info("Starting complete thresholding pipeline...")
        
        # Load all data
        data = self.load_streamline_data()
        
        if not data['true_labels']:
            self.logger.error("No true labels found - cannot proceed with thresholding")
            return {}
        
        # Apply thresholding
        thresholded_data = self.apply_thresholding(data)
        
        if not thresholded_data:
            self.logger.error("Thresholding failed")
            return {}
        
        # Save thresholded data
        saved_files = self.save_thresholded_data(thresholded_data)
        
        self.logger.info("Thresholding pipeline completed successfully!")
        return saved_files


def threshold_subject_data(subject_path: Union[str, Path], atlas: str, 
                          min_streamlines_per_node: int = 10, logger=None) -> Dict[str, str]:
    """
    Convenience function to threshold a subject's data
    
    Args:
        subject_path: Path to subject directory
        atlas: Atlas name
        min_streamlines_per_node: Minimum streamlines required per node
        logger: Logger instance (optional)
        
    Returns:
        Dictionary with paths to thresholded files
    """
    thresholder = StreamlineThresholder(subject_path, atlas, min_streamlines_per_node, logger)
    return thresholder.create_thresholded_dataset()


def main():
    """Test the thresholding system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Streamline Thresholding')
    parser.add_argument('--subject-path', '-s', required=True, help='Subject directory path')
    parser.add_argument('--atlas', '-a', required=True, help='Atlas name')
    parser.add_argument('--min-streamlines', '-m', type=int, default=10, 
                       help='Minimum streamlines per node (default: 10)')
    
    args = parser.parse_args()
    
    # Run thresholding
    print(f"Running thresholding for {args.atlas} with minimum {args.min_streamlines} streamlines per node...")
    
    saved_files = threshold_subject_data(
        subject_path=args.subject_path,
        atlas=args.atlas,
        min_streamlines_per_node=args.min_streamlines
    )
    
    if saved_files:
        print("\n✓ Thresholding completed successfully!")
        print("Saved files:")
        for key, path in saved_files.items():
            print(f"  {key}: {path}")
    else:
        print("\n✗ Thresholding failed!")


if __name__ == "__main__":
    main()