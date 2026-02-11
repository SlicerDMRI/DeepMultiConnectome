#!/usr/bin/env python3
"""
Test-Retest (TRT) Helper Functions

This module contains helper functions for loading and processing test-retest
connectome data from HCP_MRtrix_test and HCP_MRtrix_retest folders.

The connectomes are stored with the following naming conventions:
- True (traditional) NOS: output/connectome_matrix_{atlas}.csv
- True (traditional) FA: output/connectome_matrix_FA_mean_{atlas}.csv
- True (traditional) SIFT2: output/connectome_matrix_SIFT_sum_{atlas}.csv
- Predicted: TractCloud/connectome_{atlas}_{pred}.csv (if exists)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Simple cache for loaded files to avoid reloading
_file_cache = {}

def _load_cached_predictions(filepath: Path) -> Optional[np.ndarray]:
    """Load predictions file with caching."""
    cache_key = str(filepath)
    if cache_key not in _file_cache:
        try:
            data = np.loadtxt(filepath, dtype=np.int32)
            _file_cache[cache_key] = data
        except Exception:
            return None
    return _file_cache[cache_key]


def get_true_connectome_path(base_path: Path, subject_id: str, atlas: str, 
                             connectome_type: str) -> Path:
    """
    Get the path for a traditional (true) connectome file.
    
    Args:
        base_path: Base path (e.g., /media/volume/MV_HCP/HCP_MRtrix_test)
        subject_id: Subject ID
        atlas: Atlas name (aparc+aseg or aparc.a2009s+aseg)
        connectome_type: Type of connectome (nos, fa, sift2)
        
    Returns:
        Path to the connectome file
    """
    output_dir = base_path / subject_id / "output"
    
    # Map connectome types to file naming conventions
    if connectome_type == 'nos':
        filename = f"connectome_matrix_{atlas}.csv"
    elif connectome_type == 'fa':
        filename = f"connectome_matrix_FA_mean_{atlas}.csv"
    elif connectome_type == 'sift2':
        filename = f"connectome_matrix_SIFT_sum_{atlas}.csv"
    elif connectome_type == 'md':
        filename = f"connectome_matrix_MD_mean_{atlas}.csv"
    elif connectome_type == 'ad':
        filename = f"connectome_matrix_AD_mean_{atlas}.csv"
    elif connectome_type == 'rd':
        filename = f"connectome_matrix_RD_mean_{atlas}.csv"
    else:
        raise ValueError(f"Unknown connectome type: {connectome_type}")
    
    return output_dir / filename


def get_pred_connectome_path(base_path: Path, subject_id: str, atlas: str,
                             connectome_type: str) -> Optional[Path]:
    """
    Get the path for a predicted connectome file from TractCloud.
    
    Note: In the test/retest folders, predicted connectomes may use a different
    naming convention than the main HCP_MRtrix folder.
    
    Args:
        base_path: Base path (e.g., /media/volume/MV_HCP/HCP_MRtrix_test)
        subject_id: Subject ID
        atlas: Atlas name (aparc+aseg or aparc.a2009s+aseg)
        connectome_type: Type of connectome (nos, fa, sift2)
        
    Returns:
        Path to the predicted connectome file, or None if not applicable
    """
    tractcloud_dir = base_path / subject_id / "TractCloud"
    
    # Check for analysis folder first (new convention)
    analysis_dir = base_path / subject_id / "analysis" / atlas
    if analysis_dir.exists():
        filename = f"connectome_pred_{connectome_type}_{atlas}.csv"
        path = analysis_dir / filename
        if path.exists():
            return path
    
    # TractCloud predictions - check for different naming conventions
    # Convention 1: connectome_{atlas}_pred.csv (used in old TRT scripts)
    # Convention 2: Direct predictions file that needs processing
    
    # For now, we only support NOS predictions from TractCloud
    # FA and SIFT2 predictions would need to be computed from predictions
    if connectome_type == 'nos':
        # Check for existing connectome file
        filename = f"connectome_{atlas}_pred.csv"
        path = tractcloud_dir / filename
        if path.exists():
            return path
    
    return None


def load_connectome(filepath: Path, handle_nan: bool = True) -> Optional[np.ndarray]:
    """
    Load a connectome matrix from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        handle_nan: If True, replace NaN values with 0
        
    Returns:
        Numpy array of the connectome matrix, or None if loading fails
    """
    if not filepath.exists():
        return None
    
    try:
        matrix = pd.read_csv(filepath, header=None).values
        
        if handle_nan:
            matrix = np.nan_to_num(matrix, nan=0.0)
        
        return matrix
    except Exception as e:
        logging.warning(f"Error loading {filepath}: {e}")
        return None


def load_trt_subject_connectomes(
    subject_id: str,
    test_base: Path,
    retest_base: Path,
    atlases: List[str],
    connectome_types: List[str],
    include_predicted: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Load all connectomes for a subject from both test and retest sessions.
    
    Args:
        subject_id: Subject ID
        test_base: Base path to test data (e.g., /media/volume/MV_HCP/HCP_MRtrix_test)
        retest_base: Base path to retest data
        atlases: List of atlas names
        connectome_types: List of connectome types (nos, fa, sift2)
        include_predicted: Whether to include predicted connectomes
        logger: Optional logger
        
    Returns:
        Dictionary with structure:
        {
            atlas: {
                connectome_type: {
                    'test_true': np.ndarray,
                    'retest_true': np.ndarray,
                    'test_pred': np.ndarray (optional),
                    'retest_pred': np.ndarray (optional)
                }
            }
        }
    """
    data = {}
    
    for atlas in atlases:
        data[atlas] = {}
        
        for ctype in connectome_types:
            entry = {}
            
            # Load true (traditional) connectomes
            test_true_path = get_true_connectome_path(test_base, subject_id, atlas, ctype)
            retest_true_path = get_true_connectome_path(retest_base, subject_id, atlas, ctype)
            
            test_true = load_connectome(test_true_path)
            retest_true = load_connectome(retest_true_path)
            
            if test_true is not None and retest_true is not None:
                if test_true.shape == retest_true.shape:
                    entry['test_true'] = test_true
                    entry['retest_true'] = retest_true
                else:
                    if logger:
                        logger.warning(f"Shape mismatch for {subject_id} {atlas} {ctype} true: "
                                     f"{test_true.shape} vs {retest_true.shape}")
            
            # Load predicted connectomes if requested
            if include_predicted:
                test_pred_path = get_pred_connectome_path(test_base, subject_id, atlas, ctype)
                retest_pred_path = get_pred_connectome_path(retest_base, subject_id, atlas, ctype)
                
                if test_pred_path and retest_pred_path:
                    test_pred = load_connectome(test_pred_path)
                    retest_pred = load_connectome(retest_pred_path)
                    
                    if test_pred is not None and retest_pred is not None:
                        if test_pred.shape == retest_pred.shape:
                            entry['test_pred'] = test_pred
                            entry['retest_pred'] = retest_pred
                        else:
                            if logger:
                                logger.warning(f"Shape mismatch for {subject_id} {atlas} {ctype} pred: "
                                             f"{test_pred.shape} vs {retest_pred.shape}")
            
            if entry:  # Only add if we have at least some data
                data[atlas][ctype] = entry
    
    return data


def build_predicted_connectome_from_labels(
    predictions_file: Path,
    labels_file: Path,
    atlas: str,
    symmetric: bool = True
) -> Optional[np.ndarray]:
    """
    Build a predicted connectome matrix from TractCloud prediction labels.
    
    This function reads the predictions file (containing predicted labels for each
    streamline) and builds a connectome matrix by counting connections.
    
    The predictions file format is:
    - Two space-separated integers per line: roi1 roi2
    
    Args:
        predictions_file: Path to predictions_{atlas}.txt
        labels_file: Path to labels_10M_{atlas}.txt (for matrix dimensions)
        atlas: Atlas name for determining matrix size
        symmetric: If True, make the matrix symmetric
        
    Returns:
        Numpy array of the connectome matrix, or None if loading fails
    """
    if not predictions_file.exists():
        return None
    
    try:
        # Determine matrix size from atlas
        # aparc+aseg: 84 ROIs, aparc.a2009s+aseg: 164 ROIs
        if 'a2009s' in atlas:
            n_rois = 164
        else:
            n_rois = 84
        
        # Try to load with numpy for speed (space-separated two-column format)
        try:
            data = _load_cached_predictions(predictions_file)
            if data is None or data.ndim == 1 or data.shape[1] != 2:
                return None
            
            # Filter valid ROI indices
            valid_mask = (data[:, 0] >= 0) & (data[:, 0] < n_rois) & \
                        (data[:, 1] >= 0) & (data[:, 1] < n_rois)
            data = data[valid_mask]
            
            # Build connectome using np.add.at for efficient counting
            matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
            np.add.at(matrix, (data[:, 0], data[:, 1]), 1)
            
            if symmetric:
                # Add transpose (excluding diagonal to avoid double counting)
                matrix = matrix + matrix.T - np.diag(np.diag(matrix))
            
            # Check if matrix is non-empty
            if np.sum(matrix) == 0:
                return None
            
            return matrix
            
        except Exception:
            # Fall back to line-by-line parsing for non-standard formats
            matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
            
            with open(predictions_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try different separators
                    if ' ' in line:
                        parts = line.split()
                    elif ',' in line:
                        parts = line.split(',')
                    else:
                        continue
                    
                    if len(parts) != 2:
                        continue
                    
                    try:
                        roi1, roi2 = int(parts[0]), int(parts[1])
                        if 0 <= roi1 < n_rois and 0 <= roi2 < n_rois:
                            matrix[roi1, roi2] += 1
                            if symmetric and roi1 != roi2:
                                matrix[roi2, roi1] += 1
                    except ValueError:
                        continue
            
            if np.sum(matrix) == 0:
                return None
            
            return matrix
            
    except Exception as e:
        logging.warning(f"Error building predicted connectome from {predictions_file}: {e}")
        return None
