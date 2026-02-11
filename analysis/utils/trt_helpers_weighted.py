#!/usr/bin/env python3
"""
Weighted Connectome Builder for Test-Retest Analysis

Functions to build FA and SIFT2 weighted predicted connectomes from predictions
and per-streamline weight files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

# Simple cache for loaded files to avoid reloading
_file_cache = {}

def _load_cached_file(filepath: Path, dtype) -> Optional[np.ndarray]:
    """Load a file with caching to avoid repeated disk I/O."""
    cache_key = (str(filepath), dtype)
    if cache_key not in _file_cache:
        try:
            _file_cache[cache_key] = np.loadtxt(filepath, dtype=dtype)
        except Exception as e:
            logging.warning(f"Could not load {filepath}: {e}")
            return None
    return _file_cache[cache_key]


def build_weighted_connectome_from_predictions(
    predictions_file: Path,
    weights_file: Path,
    atlas: str,
    weight_type: str = 'fa',
    symmetric: bool = True
) -> Optional[np.ndarray]:
    """
    Build a weighted predicted connectome (FA or SIFT2) from predictions and weights.
    
    Args:
        predictions_file: Path to predictions file (format: "roi1 roi2" per line)
        weights_file: Path to weights file (mean_fa_per_streamline.txt or sift2_weights.txt)
        atlas: Atlas name for determining matrix size
        weight_type: 'fa' (use mean aggregation) or 'sift2' (use sum aggregation)
        symmetric: If True, make the matrix symmetric
        
    Returns:
        Numpy array of the weighted connectome matrix, or None if loading fails
    """
    if not predictions_file.exists() or not weights_file.exists():
        return None
    
    try:
        # Determine matrix size from atlas
        if 'a2009s' in atlas:
            n_rois = 164
        else:
            n_rois = 84
        
        # Load predictions and weights using cache
        predictions = _load_cached_file(predictions_file, np.int32)
        if predictions is None or predictions.ndim == 1 or predictions.shape[1] != 2:
            return None
        
        weights = _load_cached_file(weights_file, np.float32)
        if weights is None:
            return None
        
        # Check length match
        if len(predictions) != len(weights):
            logging.warning(f"Length mismatch: {len(predictions)} predictions vs {len(weights)} weights")
            return None
        
        # Filter valid ROI indices
        valid_mask = (predictions[:, 0] >= 0) & (predictions[:, 0] < n_rois) & \
                    (predictions[:, 1] >= 0) & (predictions[:, 1] < n_rois)
        
        # Also filter out NaN and Inf weights
        valid_mask = valid_mask & np.isfinite(weights)
        
        n_invalid = len(weights) - np.sum(valid_mask)
        if n_invalid > 0:
            logging.info(f"Filtered out {n_invalid} invalid weights ({100*n_invalid/len(weights):.2f}%)")
        
        predictions = predictions[valid_mask]
        weights = weights[valid_mask]
        
        if len(predictions) == 0:
            return None
        
        # Build weighted connectome using vectorized operations
        matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
        
        roi1 = predictions[:, 0]
        roi2 = predictions[:, 1]
        
        if weight_type == 'sift2':
            # SIFT2: Sum of weights for each connection (vectorized)
            np.add.at(matrix, (roi1, roi2), weights)
            if symmetric:
                # Add symmetric entries (but don't double-count diagonal)
                off_diag = roi1 != roi2
                np.add.at(matrix, (roi2[off_diag], roi1[off_diag]), weights[off_diag])
        else:
            # FA: Mean of weights for each connection (vectorized)
            # First sum all weights
            np.add.at(matrix, (roi1, roi2), weights)
            
            # Count occurrences for each connection
            count = np.zeros((n_rois, n_rois), dtype=np.int32)
            np.add.at(count, (roi1, roi2), 1)
            
            # Divide by count to get mean (avoid division by zero)
            mask = count > 0
            matrix[mask] /= count[mask]
            
            if symmetric:
                # Copy to symmetric positions
                off_diag = roi1 != roi2
                matrix[roi2[off_diag], roi1[off_diag]] = matrix[roi1[off_diag], roi2[off_diag]]
        
        # Check if matrix is non-empty
        if np.sum(matrix) == 0:
            return None
        
        return matrix
        
    except Exception as e:
        logging.warning(f"Error building weighted predicted connectome: {e}")
        return None


def validate_connectome(matrix: np.ndarray, name: str = "Connectome") -> dict:
    """
    Validate a connectome matrix and return statistics.
    
    Args:
        matrix: The connectome matrix to validate
        name: Name for logging purposes
        
    Returns:
        Dictionary with validation statistics
    """
    if matrix is None:
        return {'valid': False, 'reason': 'Matrix is None'}
    
    stats = {
        'valid': True,
        'shape': matrix.shape,
        'is_square': matrix.shape[0] == matrix.shape[1],
        'is_symmetric': np.allclose(matrix, matrix.T, rtol=1e-5),
        'n_nonzero': np.count_nonzero(matrix),
        'n_total': matrix.size,
        'sparsity': 1.0 - (np.count_nonzero(matrix) / matrix.size),
        'min': np.min(matrix),
        'max': np.max(matrix),
        'mean': np.mean(matrix),
        'std': np.std(matrix),
        'n_nan': np.sum(np.isnan(matrix)),
        'n_inf': np.sum(np.isinf(matrix)),
    }
    
    # Check for issues
    issues = []
    if not stats['is_square']:
        issues.append(f"Not square: {stats['shape']}")
    if stats['n_nan'] > 0:
        issues.append(f"{stats['n_nan']} NaN values")
    if stats['n_inf'] > 0:
        issues.append(f"{stats['n_inf']} Inf values")
    if stats['n_nonzero'] == 0:
        issues.append("All zeros")
    if stats['min'] < 0:
        issues.append(f"Negative values (min={stats['min']})")
    
    stats['issues'] = issues
    stats['valid'] = len(issues) == 0
    
    return stats


def log_connectome_validation(stats: dict, logger: Optional[logging.Logger] = None, name: str = "Connectome"):
    """
    Log connectome validation statistics.
    
    Args:
        stats: Statistics dictionary from validate_connectome
        logger: Logger instance (uses logging.info if None)
        name: Name for logging
    """
    log_func = logger.info if logger else logging.info
    
    if not stats['valid']:
        if 'reason' in stats:
            log_func(f"{name} INVALID: {stats['reason']}")
        else:
            log_func(f"{name} INVALID: {', '.join(stats['issues'])}")
        return
    
    log_func(f"{name}: {stats['shape']}, "
            f"{stats['n_nonzero']:,}/{stats['n_total']:,} nonzero ({100*(1-stats['sparsity']):.1f}% filled), "
            f"range=[{stats['min']:.3f}, {stats['max']:.3f}], "
            f"mean={stats['mean']:.3f}±{stats['std']:.3f}, "
            f"symmetric={stats['is_symmetric']}")
