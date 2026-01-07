#!/usr/bin/env python3
"""
Shared Analysis Metrics Module

This module contains shared metric computation functions used across different
connectome analysis scripts. Ensures consistent computation of:
- Pearson correlation (with k=0 to include diagonal)
- LERM (Log-Euclidean Riemannian Metric) using matrix logarithm
- Zero masking for diffusion metrics

All functions use the same validated approach as the main analysis scripts.
"""

import numpy as np
from scipy.linalg import logm, norm
from typing import Tuple, Optional
import logging


def apply_zero_mask(true_matrix: np.ndarray, 
                   pred_matrix: np.ndarray, 
                   connectome_type: str,
                   mask_zeros: bool = True,
                   logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply zero masking for diffusion metrics (FA, MD, AD, RD)
    
    Args:
        true_matrix: Ground truth connectome matrix
        pred_matrix: Predicted connectome matrix
        connectome_type: Type of connectome (nos, fa, md, ad, rd, sift2)
        mask_zeros: If True, apply masking. Default: True
        logger: Optional logger for debug output
        
    Returns:
        Tuple of (masked_true, masked_pred, mask)
        - If masking is disabled or connectome_type is 'nos' or 'sift2', returns originals
        - Otherwise, returns matrices with zeros where either original had zeros
    """
    # Only apply masking to diffusion metrics (FA, MD, AD, RD)
    diffusion_metrics = ['fa', 'md', 'ad', 'rd']
    
    if not mask_zeros or connectome_type not in diffusion_metrics:
        # No masking - return original matrices and a mask of all True
        return true_matrix, pred_matrix, np.ones_like(true_matrix, dtype=bool)
    
    # Create mask: True where both true and pred are non-zero
    mask = (true_matrix != 0) & (pred_matrix != 0)
    
    # Create masked copies
    masked_true = true_matrix.copy()
    masked_pred = pred_matrix.copy()
    
    # Set masked entries to zero in both matrices
    masked_true[~mask] = 0
    masked_pred[~mask] = 0
    
    if logger:
        n_total = true_matrix.size
        n_masked = np.sum(~mask)
        n_both_zero = np.sum((true_matrix == 0) & (pred_matrix == 0))
        n_true_zero = np.sum((true_matrix == 0) & (pred_matrix != 0))
        n_pred_zero = np.sum((true_matrix != 0) & (pred_matrix == 0))
        
        logger.debug(
            f"Zero masking for {connectome_type}: "
            f"total={n_total}, masked={n_masked} ({100*n_masked/n_total:.1f}%), "
            f"both_zero={n_both_zero}, true_zero={n_true_zero}, pred_zero={n_pred_zero}"
        )
    
    return masked_true, masked_pred, mask


def compute_correlation(true_matrix: np.ndarray, 
                        pred_matrix: np.ndarray,
                        include_diagonal: bool = True,
                        filter_zeros: bool = True) -> float:
    """
    Compute Pearson correlation between two connectome matrices
    
    Args:
        true_matrix: Ground truth connectome matrix
        pred_matrix: Predicted connectome matrix
        include_diagonal: If True, include diagonal (k=0). If False, exclude diagonal (k=1).
                         Default: True (validated approach)
        filter_zeros: If True, exclude indices where both matrices are zero. 
                     Default: True (legacy behavior/sparse handling)
    
    Returns:
        Pearson correlation coefficient
    """
    # Get upper triangle indices (k=0 includes diagonal, k=1 excludes)
    k = 0 if include_diagonal else 1
    triu_indices = np.triu_indices_from(true_matrix, k=k)
    
    true_upper = true_matrix[triu_indices]
    pred_upper = pred_matrix[triu_indices]
    
    if filter_zeros:
        # Filter out zeros if needed (for masked matrices)
        nonzero_mask = (true_upper != 0) & (pred_upper != 0)
        
        if np.sum(nonzero_mask) > 0:
            true_upper_filtered = true_upper[nonzero_mask]
            pred_upper_filtered = pred_upper[nonzero_mask]
            
            # Compute correlation
            correlation = np.corrcoef(true_upper_filtered, pred_upper_filtered)[0, 1]
            return correlation
        else:
            return np.nan
    else:
        # Pearson on full flattened arrays (including zeros)
        # Handle constant arrays (std=0) to avoid NaNs if possible, or let corrcoef handle it
        if np.std(true_upper) == 0 or np.std(pred_upper) == 0:
            return np.nan
        return np.corrcoef(true_upper, pred_upper)[0, 1]


def compute_lerm(true_matrix: np.ndarray, 
                pred_matrix: np.ndarray,
                use_matrix_log: bool = True,
                epsilon: float = 1e-10) -> float:
    """
    Compute LERM (Log-Euclidean Riemannian Metric) between two connectome matrices
    
    Args:
        true_matrix: Ground truth connectome matrix
        pred_matrix: Predicted connectome matrix
        use_matrix_log: If True, use matrix logarithm logm (validated approach).
                       If False, use element-wise log (legacy). Default: True
        epsilon: Small value to add for numerical stability. Default: 1e-10
    
    Returns:
        LERM distance
    """
    if use_matrix_log:
        # Validated approach: Use matrix logarithm
        # Add epsilon to diagonal if it is zero, to ensure invertibility
        true_log = true_matrix.copy()
        pred_log = pred_matrix.copy()
        
        # Ensure matrices are symmetric positive definite-ish
        # Add a small regularization term to the diagonal to improve stability
        # Only if we suspect instability. But simple epsilon replaces 0s which might be everywhere.
        # If the matrix is sparse, logm is very slow/unstable.
        
        # Original:
        # true_log[true_log == 0] = epsilon
        # pred_log[pred_log == 0] = epsilon
        
        # Better: Add identity regularization
        np.fill_diagonal(true_log, true_log.diagonal() + epsilon)
        np.fill_diagonal(pred_log, pred_log.diagonal() + epsilon)
        
        # Try-catch logm
        try:
             # disp=False returns (logm, errest)
             result_true = logm(true_log, disp=False)
             result_pred = logm(pred_log, disp=False)
             
             tl = result_true[0] if isinstance(result_true, tuple) else result_true
             pl = result_pred[0] if isinstance(result_pred, tuple) else result_pred
             
             lerm = norm(pl - tl, 'fro')
        except:
             lerm = np.nan
    else:
        # Legacy approach: Element-wise log
        # Flatten matrices for comparison (exclude diagonal)
        mask = ~np.eye(true_matrix.shape[0], dtype=bool)
        true_flat = true_matrix[mask].flatten()
        pred_flat = pred_matrix[mask].flatten()
        
        # Apply element-wise log
        true_log = np.log(true_flat + epsilon)
        pred_log = np.log(pred_flat + epsilon)
        
        # Compute mean absolute difference
        lerm = np.mean(np.abs(pred_log - true_log))
    
    return lerm


def compute_connectome_metrics(true_matrix: np.ndarray,
                               pred_matrix: np.ndarray,
                               connectome_type: str = 'nos',
                               mask_zeros: bool = True,
                               logger: Optional[logging.Logger] = None) -> dict:
    """
    Compute all standard connectome comparison metrics
    
    Args:
        true_matrix: Ground truth connectome matrix
        pred_matrix: Predicted connectome matrix
        connectome_type: Type of connectome (nos, fa, md, ad, rd, sift2)
        mask_zeros: If True, apply zero masking for diffusion metrics
        logger: Optional logger for debug output
    
    Returns:
        Dictionary containing:
        - correlation: Pearson correlation (k=0, includes diagonal)
        - lerm: LERM using matrix logarithm
        - true_connections: Number of non-zero connections in true matrix
        - pred_connections: Number of non-zero connections in pred matrix
        - true_mean: Mean value of true matrix
        - pred_mean: Mean value of pred matrix
        - n_masked_entries: Number of entries masked (if masking applied)
    """
    # Apply zero masking if needed
    masked_true, masked_pred, mask = apply_zero_mask(
        true_matrix, pred_matrix, connectome_type, mask_zeros, logger
    )
    
    # Compute metrics
    correlation = compute_correlation(masked_true, masked_pred, include_diagonal=True)
    lerm = compute_lerm(masked_true, masked_pred, use_matrix_log=True)
    
    # Compute connection counts and statistics
    true_connections = np.count_nonzero(true_matrix)
    pred_connections = np.count_nonzero(pred_matrix)
    true_mean = np.mean(true_matrix)
    pred_mean = np.mean(pred_matrix)
    
    # Count masked entries
    n_masked_entries = np.sum(~mask) if mask_zeros else 0
    
    return {
        'correlation': correlation,
        'lerm': lerm,
        'true_connections': true_connections,
        'pred_connections': pred_connections,
        'true_mean': true_mean,
        'pred_mean': pred_mean,
        'n_masked_entries': n_masked_entries
    }
