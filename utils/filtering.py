"""
Filtering utilities for connectome analysis

This module provides functions for:
- Length-based filtering of streamlines
- Node-based thresholding of connectomes
- Combined filtering workflows
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import logging


def load_streamline_lengths(lengths_file: Union[str, Path], logger=None) -> np.ndarray:
    """
    Load streamline lengths from file
    
    Args:
        lengths_file: Path to streamline lengths file
        logger: Optional logger instance
        
    Returns:
        Array of streamline lengths
    """
    def _log(message: str, level: str = "info"):
        if logger:
            getattr(logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    try:
        lengths_file = Path(lengths_file)
        if not lengths_file.exists():
            raise FileNotFoundError(f"Lengths file not found: {lengths_file}")
        
        with open(lengths_file, 'r') as f:
            lines = f.readlines()
        
        # Find the data line (skip comments)
        data_line = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                data_line = line
                break
        
        if not data_line:
            raise ValueError("No data found in lengths file")
        
        # Parse values
        values = data_line.split()
        lengths = []
        
        for val in values:
            try:
                if val.lower() != 'nan':
                    lengths.append(float(val))
                else:
                    lengths.append(np.nan)
            except ValueError:
                lengths.append(np.nan)
        
        lengths_array = np.array(lengths)
        _log(f"Loaded {len(lengths_array)} streamline lengths from {lengths_file}")
        return lengths_array
        
    except Exception as e:
        _log(f"Error loading streamline lengths: {e}", "error")
        return np.array([])


def apply_length_filtering(labels: List[int], 
                          diffusion_metrics: Dict[str, np.ndarray],
                          lengths: np.ndarray,
                          min_length: float = 20.0,
                          max_length: Optional[float] = None,
                          logger=None) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, int]]:
    """
    Apply length-based filtering to streamlines
    
    Args:
        labels: List of streamline labels
        diffusion_metrics: Dictionary of metric arrays
        lengths: Array of streamline lengths
        min_length: Minimum length threshold
        max_length: Maximum length threshold (optional)
        logger: Optional logger instance
        
    Returns:
        Tuple of (filtered_labels, filtered_metrics, filter_stats)
    """
    def _log(message: str, level: str = "info"):
        if logger:
            getattr(logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    if len(labels) != len(lengths):
        _log(f"Warning: Labels length ({len(labels)}) != lengths length ({len(lengths)})", "warning")
        min_len = min(len(labels), len(lengths))
        labels = labels[:min_len]
        lengths = lengths[:min_len]
    
    # Create length mask
    length_mask = lengths >= min_length
    if max_length is not None:
        length_mask &= (lengths <= max_length)
    
    # Apply filtering
    filtered_labels = [labels[i] for i in range(len(labels)) if length_mask[i]]
    
    filtered_metrics = {}
    for metric_name, metric_values in diffusion_metrics.items():
        if len(metric_values) >= len(length_mask):
            filtered_metrics[metric_name] = metric_values[length_mask]
        else:
            _log(f"Warning: {metric_name} has insufficient data for filtering", "warning")
            filtered_metrics[metric_name] = metric_values
    
    # Statistics
    stats = {
        'original_count': len(labels),
        'filtered_count': len(filtered_labels),
        'removed_count': len(labels) - len(filtered_labels),
        'removal_percentage': (len(labels) - len(filtered_labels)) / len(labels) * 100 if len(labels) > 0 else 0,
        'min_length': min_length,
        'max_length': max_length
    }
    
    length_desc = f"≥{min_length}mm"
    if max_length is not None:
        length_desc = f"{min_length}-{max_length}mm"
    
    _log(f"Length filtering ({length_desc}):")
    _log(f"  Original streamlines: {stats['original_count']:,}")
    _log(f"  Filtered streamlines: {stats['filtered_count']:,}")
    _log(f"  Removed streamlines: {stats['removed_count']:,} ({stats['removal_percentage']:.1f}%)")
    
    return filtered_labels, filtered_metrics, stats


def apply_connectome_thresholding(connectome: np.ndarray,
                                threshold_percentage: float = 5.0,
                                min_connections: int = 5,
                                logger=None) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Apply node-based thresholding to connectome
    
    Args:
        connectome: Connectome matrix
        threshold_percentage: Percentage of weakest nodes to remove
        min_connections: Minimum number of connections for a node to keep
        logger: Optional logger instance
        
    Returns:
        Tuple of (thresholded_connectome, threshold_stats)
    """
    def _log(message: str, level: str = "info"):
        if logger:
            getattr(logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    original_shape = connectome.shape
    
    # Calculate node strengths (sum of connections for each node)
    node_strengths = np.sum(connectome, axis=1) + np.sum(connectome, axis=0)
    
    # Create masks for filtering
    # Mask 1: Nodes with sufficient connections
    connection_counts = np.sum(connectome > 0, axis=1) + np.sum(connectome > 0, axis=0)
    min_connections_mask = connection_counts >= min_connections
    
    # Mask 2: Remove weakest nodes by percentage
    num_nodes_to_remove = int(len(node_strengths) * threshold_percentage / 100)
    if num_nodes_to_remove > 0:
        weakest_node_indices = np.argsort(node_strengths)[:num_nodes_to_remove]
        strength_mask = np.ones(len(node_strengths), dtype=bool)
        strength_mask[weakest_node_indices] = False
    else:
        strength_mask = np.ones(len(node_strengths), dtype=bool)
    
    # Combine masks
    keep_mask = min_connections_mask & strength_mask
    
    # Apply thresholding
    thresholded_connectome = connectome[keep_mask][:, keep_mask]
    
    # Statistics
    stats = {
        'original_nodes': original_shape[0],
        'thresholded_nodes': thresholded_connectome.shape[0],
        'removed_nodes': original_shape[0] - thresholded_connectome.shape[0],
        'removal_percentage': (original_shape[0] - thresholded_connectome.shape[0]) / original_shape[0] * 100,
        'threshold_percentage': threshold_percentage,
        'min_connections': min_connections,
        'original_connections': np.count_nonzero(connectome),
        'thresholded_connections': np.count_nonzero(thresholded_connectome)
    }
    
    _log(f"Connectome thresholding ({threshold_percentage}% nodes, min {min_connections} connections):")
    _log(f"  Original nodes: {stats['original_nodes']}")
    _log(f"  Thresholded nodes: {stats['thresholded_nodes']}")
    _log(f"  Removed nodes: {stats['removed_nodes']} ({stats['removal_percentage']:.1f}%)")
    _log(f"  Original connections: {stats['original_connections']:,}")
    _log(f"  Thresholded connections: {stats['thresholded_connections']:,}")
    
    return thresholded_connectome, stats


def create_filtered_connectome_name(base_name: str,
                                   length_filtered: bool = False,
                                   min_length: Optional[float] = None,
                                   max_length: Optional[float] = None,
                                   thresholded: bool = False,
                                   threshold_percentage: Optional[float] = None,
                                   min_connections: Optional[int] = None) -> str:
    """
    Create descriptive name for filtered connectome
    
    Args:
        base_name: Base connectome name
        length_filtered: Whether length filtering was applied
        min_length: Minimum length threshold
        max_length: Maximum length threshold
        thresholded: Whether node thresholding was applied
        threshold_percentage: Threshold percentage
        min_connections: Minimum connections threshold
        
    Returns:
        Descriptive connectome name
    """
    name_parts = [base_name]
    
    if length_filtered:
        if min_length is not None:
            if max_length is not None:
                name_parts.append(f"length_{min_length}-{max_length}mm")
            else:
                name_parts.append(f"length_min{min_length}mm")
    
    if thresholded:
        threshold_parts = []
        if threshold_percentage is not None:
            threshold_parts.append(f"{threshold_percentage}pct")
        if min_connections is not None:
            threshold_parts.append(f"min{min_connections}conn")
        if threshold_parts:
            name_parts.append(f"thresh_{'_'.join(threshold_parts)}")
    
    return "_".join(name_parts)


def save_filtering_report(filter_stats: Dict[str, Dict],
                         output_path: Union[str, Path],
                         logger=None) -> str:
    """
    Save comprehensive filtering report
    
    Args:
        filter_stats: Dictionary of filtering statistics
        output_path: Output directory path
        logger: Optional logger instance
        
    Returns:
        Path to saved report
    """
    def _log(message: str, level: str = "info"):
        if logger:
            getattr(logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary DataFrame
    summary_data = []
    
    for filter_name, stats in filter_stats.items():
        summary_data.append({
            'Filter_Type': filter_name,
            'Original_Count': stats.get('original_count', stats.get('original_nodes', 'N/A')),
            'Filtered_Count': stats.get('filtered_count', stats.get('thresholded_nodes', 'N/A')),
            'Removed_Count': stats.get('removed_count', stats.get('removed_nodes', 'N/A')),
            'Removal_Percentage': stats.get('removal_percentage', 'N/A'),
            'Parameters': _format_filter_parameters(stats)
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save report
    report_path = output_path / "filtering_report.csv"
    df.to_csv(report_path, index=False)
    
    # Save detailed stats
    detailed_path = output_path / "filtering_detailed_stats.csv"
    detailed_data = []
    
    for filter_name, stats in filter_stats.items():
        for key, value in stats.items():
            detailed_data.append({
                'Filter_Type': filter_name,
                'Statistic': key,
                'Value': value
            })
    
    pd.DataFrame(detailed_data).to_csv(detailed_path, index=False)
    
    _log(f"Filtering report saved to {report_path}")
    _log(f"Detailed filtering stats saved to {detailed_path}")
    
    return str(report_path)


def _format_filter_parameters(stats: Dict) -> str:
    """Format filter parameters for display"""
    params = []
    
    if 'min_length' in stats:
        if stats.get('max_length'):
            params.append(f"length: {stats['min_length']}-{stats['max_length']}mm")
        else:
            params.append(f"min_length: {stats['min_length']}mm")
    
    if 'threshold_percentage' in stats:
        params.append(f"threshold: {stats['threshold_percentage']}%")
    
    if 'min_connections' in stats:
        params.append(f"min_conn: {stats['min_connections']}")
    
    return "; ".join(params) if params else "None"