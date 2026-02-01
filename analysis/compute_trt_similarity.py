#!/usr/bin/env python3
"""
Compute Test-Retest Connectome Similarity (Reproducibility Study)

This script computes test-retest reliability metrics by comparing each subject's
test and retest connectomes. Results are aggregated across subjects to produce
mean ± std reproducibility statistics.

Workflow:
1. For each subject: Compute similarity between their test and retest connectomes
2. Aggregate: Compute mean ± std across all subjects
3. Compare: True (traditional) vs Predicted (TractCloud) connectomes

Metrics:
- Pearson Correlation (r): Measures linear relationship
- LERM (Log-Euclidean Riemannian Metric): Riemannian distance

Categories:
- Atlases: aparc+aseg (84 ROIs), aparc.a2009s+aseg (164 ROIs)
- Weights: NOS (streamline count), FA (mean FA), SIFT2 (SIFT2 sum)
- Versions: True (traditional), Predicted (TractCloud)

Output Files:
- trt_similarity_results.csv: Per-subject TRT metrics (one row per subject)
- summary_statistics.csv: Mean ± std across subjects (for each atlas/type/version)
- plots/: Visualization of TRT reliability

Usage:
    python3 compute_trt_similarity.py --subjects_file /path/to/subjects.txt
    python3 compute_trt_similarity.py --no_diagonal  # Exclude diagonal from calculations
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.linalg import logm, norm
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from statannotations.Annotator import Annotator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): 
        return iterable

# Add path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(Path("/media/volume/HCP_diffusion_MV/DeepMultiConnectome")))

# Import shared analysis metrics
from analysis.utils.analysis_metrics import (
    compute_correlation,
    compute_lerm
)
from analysis.utils.trt_helpers import (
    load_trt_subject_connectomes,
    get_true_connectome_path,
    get_pred_connectome_path,
    load_connectome,
    build_predicted_connectome_from_labels
)
from analysis.utils.trt_helpers_weighted import (
    build_weighted_connectome_from_predictions,
    validate_connectome,
    log_connectome_validation
)


def save_connectome(matrix: np.ndarray, filepath: Path) -> bool:
    """Save a connectome matrix to CSV file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matrix).to_csv(filepath, header=False, index=False)
        return True
    except Exception as e:
        return False


def get_pred_connectome_filename(atlas: str, ctype: str) -> str:
    """Get the filename for a predicted connectome."""
    return f"connectome_pred_{ctype}_{atlas}.csv"

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Font settings
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


class TRTSimilarityAnalysis:
    """
    Computes test-retest similarity metrics for traditional and predicted connectomes.
    """
    
    def __init__(self, 
                 subject_list_file: str,
                 test_base_path: str = "/media/volume/MV_HCP/HCP_MRtrix_test",
                 retest_base_path: str = "/media/volume/MV_HCP/HCP_MRtrix_retest",
                 max_subjects: int = None,
                 no_diagonal: bool = False,
                 atlases: List[str] = None,
                 connectome_types: List[str] = None):
        """
        Initialize the TRT analysis.
        
        Args:
            subject_list_file: Path to file with subject IDs (one per line)
            test_base_path: Base path to test data
            retest_base_path: Base path to retest data
            max_subjects: Maximum number of subjects to process (None for all)
            no_diagonal: If True, exclude diagonal from calculations
            atlases: List of atlases to process
            connectome_types: List of connectome types to process
        """
        self.subject_list_file = Path(subject_list_file)
        self.test_base = Path(test_base_path)
        self.retest_base = Path(retest_base_path)
        self.max_subjects = max_subjects
        self.no_diagonal = no_diagonal
        
        # Setup output directories
        results_dir_name = "trt_results"
        if self.no_diagonal:
            results_dir_name += "_nodiagonal"
            
        self.results_dir = Path("/media/volume/HCP_diffusion_MV/DeepMultiConnectome/analysis") / results_dir_name
        self.cache_dir = self.results_dir / "cache"
        self.plots_dir = self.results_dir / "plots"
        
        for d in [self.results_dir, self.cache_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Load subjects
        self.subjects = self._load_subject_list()
        
        # Config
        self.atlases = atlases if atlases else ["aparc+aseg", "aparc.a2009s+aseg"]
        self.connectome_types = connectome_types if connectome_types else ["nos", "fa", "sift2"]
        
        self.log(f"Initialized TRT analysis for {len(self.subjects)} subjects")
        self.log(f"Test path: {self.test_base}")
        self.log(f"Retest path: {self.retest_base}")
        self.log(f"Atlases: {self.atlases}")
        self.log(f"Connectome types: {self.connectome_types}")
        self.log(f"Exclude diagonal: {self.no_diagonal}")
        self.log(f"Results directory: {self.results_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup a proper logger that writes to file and stdout"""
        log_file = self.results_dir / f"trt_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger("TRTSimilarity")
        logger.setLevel(logging.INFO)
        logger.handlers = []  # Clear existing handlers
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def log(self, message: str):
        """Log message to file and console"""
        self.logger.info(message)
    
    def _load_subject_list(self) -> List[str]:
        """Load subject IDs from file"""
        if not self.subject_list_file.exists():
            raise FileNotFoundError(f"Subject list not found: {self.subject_list_file}")
        
        with open(self.subject_list_file, 'r') as f:
            subjects = [line.strip() for line in f if line.strip()]
        
        if self.max_subjects is not None:
            subjects = subjects[:self.max_subjects]
            self.log(f"Limited to first {self.max_subjects} subjects")
        
        return subjects
    
    def load_subject_connectomes_trt(self, subject_id: str) -> Dict:
        """
        Load connectomes for a subject from both test and retest folders.
        
        Returns a dictionary with structure:
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
        
        # Create task list for progress tracking
        tasks = [(atlas, ctype, session) for atlas in self.atlases 
                 for ctype in self.connectome_types 
                 for session in ['test', 'retest']]
        
        for atlas in self.atlases:
            data[atlas] = {}
            
            for ctype in self.connectome_types:
                entry = {}
                
                # Load TRUE connectomes from output folder
                test_true_path = get_true_connectome_path(self.test_base, subject_id, atlas, ctype)
                retest_true_path = get_true_connectome_path(self.retest_base, subject_id, atlas, ctype)
                
                test_true = load_connectome(test_true_path)
                retest_true = load_connectome(retest_true_path)
                
                if test_true is not None and retest_true is not None:
                    if test_true.shape == retest_true.shape:
                        entry['test_true'] = test_true
                        entry['retest_true'] = retest_true
                
                # Load PREDICTED connectomes with progress indication
                test_pred = self._load_or_build_predicted_connectome(
                    self.test_base, subject_id, atlas, ctype, session='test'
                )
                retest_pred = self._load_or_build_predicted_connectome(
                    self.retest_base, subject_id, atlas, ctype, session='retest'
                )
                
                if test_pred is not None and retest_pred is not None:
                    if test_pred.shape == retest_pred.shape:
                        entry['test_pred'] = test_pred
                        entry['retest_pred'] = retest_pred
                
                if entry:
                    data[atlas][ctype] = entry
        
        return data
    
    def _load_or_build_predicted_connectome(self, base_path: Path, subject_id: str, 
                                             atlas: str, ctype: str, session: str = '') -> Optional[np.ndarray]:
        """
        Load a predicted connectome, building from predictions if necessary.
        
        For NOS: Build from predictions file (streamline counts)
        For FA/SIFT2: Build weighted connectomes from predictions + weight files
        """
        # Check for existing predicted connectome file (old format from TractCloud folder)
        tractcloud_dir = base_path / subject_id / "TractCloud"
        
        # Try old naming convention: connectome_{atlas}_pred.csv
        old_format_path = tractcloud_dir / f"connectome_{atlas}_pred.csv"
        if old_format_path.exists():
            mat = load_connectome(old_format_path)
            if mat is not None and ctype == 'nos':
                stats = validate_connectome(mat, f"{subject_id} pred NOS {atlas}")
                if not stats['valid']:
                    self.log(f"    WARNING: Loaded pred NOS {atlas} is invalid: {', '.join(stats['issues'])}")
            return mat
        
        # Check analysis folder for new convention
        analysis_dir = base_path / subject_id / "analysis" / atlas
        if analysis_dir.exists():
            filename = f"connectome_pred_{ctype}_{atlas}.csv"
            path = analysis_dir / filename
            if path.exists():
                return load_connectome(path)
        
        # Build from predictions
        predictions_file = tractcloud_dir / f"predictions_{atlas}.txt"
        if not predictions_file.exists():
            return None
        
        dmri_dir = base_path / subject_id / "dMRI"
        
        # Build based on connectome type with progress indicator
        desc = f"Building {subject_id} {ctype.upper()} {atlas}{session}"
        
        # Build based on connectome type
        if ctype == 'nos':
            # NOS: Count of streamlines (unweighted)
            import time
            start = time.time()
            
            with tqdm(total=1, desc=desc, leave=False, file=sys.stdout) as pbar:
                mat = build_predicted_connectome_from_labels(
                    predictions_file, None, atlas, symmetric=True
                )
                pbar.update(1)
            
            elapsed = time.time() - start
            
            if mat is not None:
                # Validate
                stats = validate_connectome(mat, f"{subject_id} pred NOS {atlas}")
                if stats['valid']:
                    self.log(f"    Built pred NOS {atlas}{session} in {elapsed:.1f}s: {stats['n_nonzero']:,} connections")
                    # Save the built connectome for future reuse
                    analysis_dir = base_path / subject_id / "analysis" / atlas
                    save_path = analysis_dir / get_pred_connectome_filename(atlas, ctype)
                    if save_connectome(mat, save_path):
                        self.log(f"    Saved to {save_path}")
                else:
                    self.log(f"    Built pred NOS {atlas}{session} but INVALID: {', '.join(stats['issues'])}")
                    return None
            return mat
            
        elif ctype == 'fa':
            # FA: Weighted by mean FA per streamline
            fa_file = dmri_dir / "mean_fa_per_streamline.txt"
            if not fa_file.exists():
                return None
            
            import time
            start = time.time()
            
            with tqdm(total=1, desc=desc, leave=False, file=sys.stdout) as pbar:
                mat = build_weighted_connectome_from_predictions(
                    predictions_file, fa_file, atlas, weight_type='fa', symmetric=True
                )
                pbar.update(1)
            
            elapsed = time.time() - start
            
            if mat is not None:
                # Validate
                stats = validate_connectome(mat, f"{subject_id} pred FA {atlas}")
                if stats['valid']:
                    self.log(f"    Built pred FA {atlas}{session} in {elapsed:.1f}s: {stats['n_nonzero']:,} connections, mean={stats['mean']:.3f}")
                    # Save the built connectome for future reuse
                    analysis_dir = base_path / subject_id / "analysis" / atlas
                    save_path = analysis_dir / get_pred_connectome_filename(atlas, ctype)
                    if save_connectome(mat, save_path):
                        self.log(f"    Saved to {save_path}")
                else:
                    self.log(f"    Built pred FA {atlas}{session} but INVALID: {', '.join(stats['issues'])}")
                    return None
            return mat
            
        elif ctype == 'sift2':
            # SIFT2: Weighted by SIFT2 weights (sum)
            sift_file = dmri_dir / "sift2_weights.txt"
            if not sift_file.exists():
                return None
            
            import time
            start = time.time()
            
            with tqdm(total=1, desc=desc, leave=False, file=sys.stdout) as pbar:
                mat = build_weighted_connectome_from_predictions(
                    predictions_file, sift_file, atlas, weight_type='sift2', symmetric=True
                )
                pbar.update(1)
            
            elapsed = time.time() - start
            
            if mat is not None:
                # Validate
                stats = validate_connectome(mat, f"{subject_id} pred SIFT2 {atlas}")
                if stats['valid']:
                    self.log(f"    Built pred SIFT2 {atlas}{session} in {elapsed:.1f}s: {stats['n_nonzero']:,} connections, sum={stats['mean']*stats['n_nonzero']:.1f}")
                    # Save the built connectome for future reuse
                    analysis_dir = base_path / subject_id / "analysis" / atlas
                    save_path = analysis_dir / get_pred_connectome_filename(atlas, ctype)
                    if save_connectome(mat, save_path):
                        self.log(f"    Saved to {save_path}")
                else:
                    self.log(f"    Built pred SIFT2 {atlas}{session} but INVALID: {', '.join(stats['issues'])}")
                    return None
            return mat
        
        return None
    
    def compute_similarity_metrics(self, mat1: np.ndarray, mat2: np.ndarray) -> Dict:
        """
        Compute Pearson's r and LERM between two connectome matrices.
        
        Args:
            mat1: First connectome matrix
            mat2: Second connectome matrix
            
        Returns:
            Dictionary with 'pearson_r' and 'lerm_dist'
        """
        include_diagonal = not self.no_diagonal
        
        # Pearson correlation
        pearson_r = compute_correlation(mat1, mat2, include_diagonal=include_diagonal, filter_zeros=False)
        
        # LERM distance
        lerm_dist = compute_lerm(mat1, mat2, use_matrix_log=True)
        
        return {
            'pearson_r': pearson_r,
            'lerm_dist': lerm_dist
        }
    
    def compute_all_trt_similarities_vectorized(self, all_subjects_data: Dict) -> List[Dict]:
        """
        Compute test-retest similarities for all subjects using VECTORIZED operations.
        This is much faster than computing each pair individually.
        
        Uses the same vectorized approach as compute_connectome_similarities.py:
        - Z-score normalization for fast Pearson correlation
        - Batched matrix logarithm for LERM
        """
        results = []
        subjects = list(all_subjects_data.keys())
        n_subs = len(subjects)
        
        tasks = [(a, c) for a in self.atlases for c in self.connectome_types]
        
        for atlas, ctype in tqdm(tasks, desc="Computing TRT Metrics (Vectorized)", file=sys.stdout):
            # Collect valid subjects and their matrices
            valid_subs = []
            test_true_mats = []
            retest_true_mats = []
            test_pred_mats = []
            retest_pred_mats = []
            
            # Check dimensions from first valid subject
            dim = None
            for sub in subjects:
                if atlas in all_subjects_data[sub] and ctype in all_subjects_data[sub][atlas]:
                    entry = all_subjects_data[sub][atlas][ctype]
                    if 'test_true' in entry:
                        dim = entry['test_true'].shape[0]
                        break
            
            if dim is None:
                continue
            
            # Get upper triangle indices
            k = 1 if self.no_diagonal else 0
            triu_idx = np.triu_indices(dim, k=k)
            
            # Collect matrices for all valid subjects
            for sub in subjects:
                if atlas not in all_subjects_data[sub]:
                    continue
                if ctype not in all_subjects_data[sub][atlas]:
                    continue
                
                entry = all_subjects_data[sub][atlas][ctype]
                has_true = 'test_true' in entry and 'retest_true' in entry
                has_pred = 'test_pred' in entry and 'retest_pred' in entry
                
                if has_true or has_pred:
                    valid_subs.append(sub)
                    
                    if has_true:
                        test_true_mats.append(entry['test_true'][triu_idx])
                        retest_true_mats.append(entry['retest_true'][triu_idx])
                    else:
                        test_true_mats.append(None)
                        retest_true_mats.append(None)
                    
                    if has_pred:
                        test_pred_mats.append(entry['test_pred'][triu_idx])
                        retest_pred_mats.append(entry['retest_pred'][triu_idx])
                    else:
                        test_pred_mats.append(None)
                        retest_pred_mats.append(None)
            
            if not valid_subs:
                continue
            
            n_valid = len(valid_subs)
            
            # Helper: Z-score normalization for vectorized Pearson
            def z_score(arr):
                means = arr.mean(axis=1, keepdims=True)
                stds = arr.std(axis=1, keepdims=True)
                return (arr - means) / (stds + 1e-10)
            
            # Helper: Compute logm for LERM
            def get_logm_flat(mat, epsilon=1e-10):
                m = mat.copy()
                np.fill_diagonal(m, m.diagonal() + epsilon)
                try:
                    res = logm(m, disp=False)
                    lm = res[0] if isinstance(res, tuple) else res
                    return np.real(lm).ravel()
                except:
                    return np.full(mat.size, np.nan)
            
            # Vectorized Pearson for TRUE
            true_mask = [t is not None for t in test_true_mats]
            if any(true_mask):
                test_true_arr = np.array([t for t in test_true_mats if t is not None])
                retest_true_arr = np.array([r for r in retest_true_mats if r is not None])
                
                test_z = z_score(test_true_arr)
                retest_z = z_score(retest_true_arr)
                
                # Row-wise dot product gives correlation
                true_r_vals = np.sum(test_z * retest_z, axis=1) / test_true_arr.shape[1]
                
                # Compute LERM for TRUE
                true_lerm_vals = []
                true_idx = 0
                for i, sub in enumerate(valid_subs):
                    if true_mask[i]:
                        entry = all_subjects_data[sub][atlas][ctype]
                        test_log = get_logm_flat(entry['test_true'])
                        retest_log = get_logm_flat(entry['retest_true'])
                        if not np.isnan(test_log).any() and not np.isnan(retest_log).any():
                            lerm = np.linalg.norm(test_log - retest_log)
                        else:
                            lerm = np.nan
                        true_lerm_vals.append(lerm)
                        true_idx += 1
                
                # Store TRUE results
                true_idx = 0
                for i, sub in enumerate(valid_subs):
                    if true_mask[i]:
                        results.append({
                            'subject_id': sub,
                            'atlas': atlas,
                            'connectome_type': ctype,
                            'version': 'true',
                            'pearson_r': true_r_vals[true_idx],
                            'lerm_dist': true_lerm_vals[true_idx]
                        })
                        true_idx += 1
            
            # Vectorized Pearson for PRED
            pred_mask = [p is not None for p in test_pred_mats]
            if any(pred_mask):
                test_pred_arr = np.array([t for t in test_pred_mats if t is not None])
                retest_pred_arr = np.array([r for r in retest_pred_mats if r is not None])
                
                test_z = z_score(test_pred_arr)
                retest_z = z_score(retest_pred_arr)
                
                # Row-wise dot product gives correlation
                pred_r_vals = np.sum(test_z * retest_z, axis=1) / test_pred_arr.shape[1]
                
                # Compute LERM for PRED
                pred_lerm_vals = []
                pred_idx = 0
                for i, sub in enumerate(valid_subs):
                    if pred_mask[i]:
                        entry = all_subjects_data[sub][atlas][ctype]
                        test_log = get_logm_flat(entry['test_pred'])
                        retest_log = get_logm_flat(entry['retest_pred'])
                        if not np.isnan(test_log).any() and not np.isnan(retest_log).any():
                            lerm = np.linalg.norm(test_log - retest_log)
                        else:
                            lerm = np.nan
                        pred_lerm_vals.append(lerm)
                        pred_idx += 1
                
                # Store PRED results
                pred_idx = 0
                for i, sub in enumerate(valid_subs):
                    if pred_mask[i]:
                        results.append({
                            'subject_id': sub,
                            'atlas': atlas,
                            'connectome_type': ctype,
                            'version': 'pred',
                            'pearson_r': pred_r_vals[pred_idx],
                            'lerm_dist': pred_lerm_vals[pred_idx]
                        })
                        pred_idx += 1
        
        return results
    
    def compute_all_trt_similarities(self, all_subjects_data: Dict) -> List[Dict]:
        """
        Compute test-retest similarities for all subjects.
        
        For each subject, atlas, and connectome type, computes:
        - True TRT: correlation between test_true and retest_true
        - Predicted TRT: correlation between test_pred and retest_pred
        """
        results = []
        subjects = list(all_subjects_data.keys())
        
        tasks = [(a, c) for a in self.atlases for c in self.connectome_types]
        
        for atlas, ctype in tqdm(tasks, desc="Computing TRT Metrics", file=sys.stdout):
            for subject_id in subjects:
                if atlas not in all_subjects_data[subject_id]:
                    continue
                if ctype not in all_subjects_data[subject_id][atlas]:
                    continue
                
                entry = all_subjects_data[subject_id][atlas][ctype]
                
                # True TRT (traditional connectomes)
                if 'test_true' in entry and 'retest_true' in entry:
                    metrics = self.compute_similarity_metrics(
                        entry['test_true'], entry['retest_true']
                    )
                    results.append({
                        'subject_id': subject_id,
                        'atlas': atlas,
                        'connectome_type': ctype,
                        'version': 'true',
                        'pearson_r': metrics['pearson_r'],
                        'lerm_dist': metrics['lerm_dist']
                    })
                
                # Predicted TRT (TractCloud predictions)
                if 'test_pred' in entry and 'retest_pred' in entry:
                    metrics = self.compute_similarity_metrics(
                        entry['test_pred'], entry['retest_pred']
                    )
                    results.append({
                        'subject_id': subject_id,
                        'atlas': atlas,
                        'connectome_type': ctype,
                        'version': 'pred',
                        'pearson_r': metrics['pearson_r'],
                        'lerm_dist': metrics['lerm_dist']
                    })
        
        return results
    
    def compute_intersubject_similarities(self, all_subjects_data: Dict) -> List[Dict]:
        """
        Compute intersubject similarities within each session (test-test and retest-retest).
        This provides a baseline to compare TRT reliability against.
        """
        results = []
        subjects = list(all_subjects_data.keys())
        n_subs = len(subjects)
        
        if n_subs < 2:
            return results
        
        tasks = [(a, c) for a in self.atlases for c in self.connectome_types]
        
        for atlas, ctype in tqdm(tasks, desc="Computing Intersubject Metrics", file=sys.stdout):
            # Collect matrices for valid subjects
            test_true_matrices = {}
            retest_true_matrices = {}
            test_pred_matrices = {}
            retest_pred_matrices = {}
            
            for subject_id in subjects:
                if atlas not in all_subjects_data[subject_id]:
                    continue
                if ctype not in all_subjects_data[subject_id][atlas]:
                    continue
                
                entry = all_subjects_data[subject_id][atlas][ctype]
                
                if 'test_true' in entry:
                    test_true_matrices[subject_id] = entry['test_true']
                if 'retest_true' in entry:
                    retest_true_matrices[subject_id] = entry['retest_true']
                if 'test_pred' in entry:
                    test_pred_matrices[subject_id] = entry['test_pred']
                if 'retest_pred' in entry:
                    retest_pred_matrices[subject_id] = entry['retest_pred']
            
            # Compute intersubject metrics
            for version, matrices in [('true_test', test_true_matrices), 
                                       ('true_retest', retest_true_matrices),
                                       ('pred_test', test_pred_matrices),
                                       ('pred_retest', retest_pred_matrices)]:
                
                subs = list(matrices.keys())
                if len(subs) < 2:
                    continue
                
                # Compute all pairwise correlations
                pearson_vals = []
                lerm_vals = []
                
                for i, sub1 in enumerate(subs):
                    for j, sub2 in enumerate(subs):
                        if i >= j:  # Only upper triangle
                            continue
                        
                        metrics = self.compute_similarity_metrics(
                            matrices[sub1], matrices[sub2]
                        )
                        pearson_vals.append(metrics['pearson_r'])
                        lerm_vals.append(metrics['lerm_dist'])
                
                # Store mean intersubject similarity
                results.append({
                    'atlas': atlas,
                    'connectome_type': ctype,
                    'comparison_type': f'intersubject_{version}',
                    'pearson_r_mean': np.nanmean(pearson_vals),
                    'pearson_r_std': np.nanstd(pearson_vals),
                    'lerm_dist_mean': np.nanmean(lerm_vals),
                    'lerm_dist_std': np.nanstd(lerm_vals),
                    'n_pairs': len(pearson_vals)
                })
        
        return results
    
    def run_analysis(self):
        """Main execution flow."""
        self.log("=" * 60)
        self.log("Starting Test-Retest Similarity Analysis")
        self.log("=" * 60)
        
        # 1. Load all subject connectomes
        self.log(f"\nLoading connectomes for {len(self.subjects)} subjects...")
        all_subjects_data = {}
        
        for subject_id in tqdm(self.subjects, desc="Loading Subjects", file=sys.stdout):
            try:
                data = self.load_subject_connectomes_trt(subject_id)
                if data:
                    all_subjects_data[subject_id] = data
            except Exception as e:
                self.log(f"ERROR loading {subject_id}: {e}")
        
        if not all_subjects_data:
            self.log("ERROR: No subjects loaded successfully!")
            return
        
        self.log(f"\nSuccessfully loaded {len(all_subjects_data)} subjects")
        
        # 2. Compute TRT similarities using VECTORIZED operations (much faster)
        self.log("\nComputing test-retest similarity metrics (vectorized)...")
        trt_results = self.compute_all_trt_similarities_vectorized(all_subjects_data)
        
        if not trt_results:
            self.log("ERROR: No TRT results computed!")
            return
        
        # Save TRT results
        df_trt = pd.DataFrame(trt_results)
        out_csv = self.results_dir / "trt_similarity_results.csv"
        df_trt.to_csv(out_csv, index=False)
        self.log(f"TRT results saved to {out_csv}")
        
        # 3. Create summary statistics (mean ± std across subjects)
        self.create_summary_statistics(df_trt)
        
        # 4. Compute significance tests (comparing true vs predicted TRT)
        self.compute_significance_tests(df_trt)
        
        # 5. Create plots
        self.create_summary_plots(df_trt)
        
        self.log("\n" + "=" * 60)
        self.log("Analysis Complete!")
        self.log("=" * 60)
    
    def create_summary_statistics(self, df: pd.DataFrame):
        """Generate summary statistics for TRT results."""
        self.log("\nGenerating summary statistics...")
        
        summary = df.groupby(['atlas', 'connectome_type', 'version']).agg({
            'pearson_r': ['mean', 'std', 'count'],
            'lerm_dist': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        out_summary = self.results_dir / "summary_statistics.csv"
        summary.to_csv(out_summary, index=False)
        self.log(f"Summary statistics saved to {out_summary}")
        
        # Print summary
        self.log("\n" + "-" * 60)
        self.log("TRT Summary Statistics:")
        self.log("-" * 60)
        for _, row in summary.iterrows():
            atlas_label = "84 ROIs" if "aparc+aseg" == row['atlas'] else "164 ROIs"
            self.log(f"{atlas_label} | {row['connectome_type'].upper()} | {row['version']} | "
                    f"r={row['pearson_r_mean']:.3f}±{row['pearson_r_std']:.3f}, "
                    f"LERM={row['lerm_dist_mean']:.2f}±{row['lerm_dist_std']:.2f} "
                    f"(n={row['pearson_r_count']:.0f})")
    
    def create_trt_vs_intersubject_comparison(self, df_trt: pd.DataFrame, inter_results: List[Dict]):
        """Create a comparison table of TRT vs Intersubject similarities."""
        if not inter_results:
            return
        
        self.log("\nGenerating TRT vs Intersubject comparison...")
        
        # Compute TRT summary
        trt_summary = df_trt.groupby(['atlas', 'connectome_type', 'version']).agg({
            'pearson_r': ['mean', 'std'],
            'lerm_dist': ['mean', 'std']
        }).reset_index()
        trt_summary.columns = ['atlas', 'connectome_type', 'version', 
                               'trt_r_mean', 'trt_r_std', 'trt_lerm_mean', 'trt_lerm_std']
        
        # Convert inter results to comparison format
        comparison_rows = []
        for inter in inter_results:
            comparison_rows.append({
                'atlas': inter['atlas'],
                'connectome_type': inter['connectome_type'],
                'comparison': inter['comparison_type'],
                'inter_r_mean': inter['pearson_r_mean'],
                'inter_r_std': inter['pearson_r_std'],
                'inter_lerm_mean': inter['lerm_dist_mean'],
                'inter_lerm_std': inter['lerm_dist_std']
            })
        
        df_comparison = pd.DataFrame(comparison_rows)
        out_comparison = self.results_dir / "trt_vs_intersubject_comparison.csv"
        df_comparison.to_csv(out_comparison, index=False)
        self.log(f"Comparison saved to {out_comparison}")
    
    def compute_significance_tests(self, df: pd.DataFrame):
        """Compute significance tests comparing true vs predicted TRT."""
        self.log("\nComputing significance tests...")
        out_file = self.results_dir / "significance_tests.txt"
        
        with open(out_file, 'w') as f:
            f.write("Test-Retest Similarity Significance Tests\n")
            f.write("=" * 60 + "\n\n")
            f.write("Comparing True vs Predicted TRT similarity (Wilcoxon signed-rank test)\n\n")
            
            for atlas in self.atlases:
                atlas_label = "84 ROIs" if atlas == "aparc+aseg" else "164 ROIs"
                f.write(f"\n{atlas_label} ({atlas})\n")
                f.write("-" * 40 + "\n")
                
                for ctype in self.connectome_types:
                    df_sub = df[(df['atlas'] == atlas) & (df['connectome_type'] == ctype)]
                    
                    true_r = df_sub[df_sub['version'] == 'true']['pearson_r'].dropna().values
                    pred_r = df_sub[df_sub['version'] == 'pred']['pearson_r'].dropna().values
                    
                    if len(true_r) > 5 and len(pred_r) > 5:
                        # Match subjects for paired test
                        true_subs = df_sub[df_sub['version'] == 'true'].set_index('subject_id')['pearson_r']
                        pred_subs = df_sub[df_sub['version'] == 'pred'].set_index('subject_id')['pearson_r']
                        common_subs = true_subs.index.intersection(pred_subs.index)
                        
                        if len(common_subs) > 5:
                            true_paired = true_subs.loc[common_subs].values
                            pred_paired = pred_subs.loc[common_subs].values
                            
                            try:
                                stat, p_val = wilcoxon(true_paired, pred_paired)
                                f.write(f"\n{ctype.upper()} Pearson's r (True vs Pred):\n")
                                f.write(f"  True: {np.mean(true_paired):.4f} ± {np.std(true_paired):.4f}\n")
                                f.write(f"  Pred: {np.mean(pred_paired):.4f} ± {np.std(pred_paired):.4f}\n")
                                f.write(f"  Wilcoxon W={stat:.2f}, p={p_val:.4e}\n")
                            except Exception as e:
                                f.write(f"\n{ctype.upper()}: Could not compute test for Pearson's r: {e}\n")

                            # LERM distance
                            true_lerm_subs = df_sub[df_sub['version'] == 'true'].set_index('subject_id')['lerm_dist']
                            pred_lerm_subs = df_sub[df_sub['version'] == 'pred'].set_index('subject_id')['lerm_dist']
                            common_lerm_subs = true_lerm_subs.index.intersection(pred_lerm_subs.index)

                            if len(common_lerm_subs) > 5:
                                true_lerm_paired = true_lerm_subs.loc[common_lerm_subs].values
                                pred_lerm_paired = pred_lerm_subs.loc[common_lerm_subs].values

                                try:
                                    stat, p_val = wilcoxon(true_lerm_paired, pred_lerm_paired)
                                    f.write(f"\n{ctype.upper()} LERM Distance (True vs Pred):\n")
                                    f.write(f"  True: {np.mean(true_lerm_paired):.4f} ± {np.std(true_lerm_paired):.4f}\n")
                                    f.write(f"  Pred: {np.mean(pred_lerm_paired):.4f} ± {np.std(pred_lerm_paired):.4f}\n")
                                    f.write(f"  Wilcoxon W={stat:.2f}, p={p_val:.4e}\n")
                                except Exception as e:
                                    f.write(f"\n{ctype.upper()}: Could not compute test for LERM: {e}\n")

                    else:
                        f.write(f"\n{ctype.upper()}: Insufficient data for comparison\n")
        
        self.log(f"Significance tests saved to {out_file}")
    
    def create_summary_plots(self, df: pd.DataFrame):
        """Create summary visualization plots with significance annotations."""
        self.log("\nGenerating summary plots...")

        # Map atlas names for display
        atlas_map = {
            'aparc+aseg': '84 ROIs',
            'aparc.a2009s+aseg': '164 ROIs'
        }
        df_plot = df.copy()
        df_plot['ROI Count'] = df_plot['atlas'].map(atlas_map)
        df_plot['Version'] = df_plot['version'].map({'true': 'Traditional', 'pred': 'Predicted'})
        
        # Map connectome types to weighted labels
        ctype_map = {
            'nos': 'NoS-weighted',
            'fa': 'FA-weighted',
            'sift2': 'SIFT2-weighted'
        }
        df_plot['Connectome Type'] = df_plot['connectome_type'].map(ctype_map)
        
        # Setup style
        font_settings = {'font.family': 'DejaVu Sans', 'font.sans-serif': ['DejaVu Sans']}
        sns.set_context("notebook", font_scale=1.2, rc=font_settings)
        sns.set_style("whitegrid", rc=font_settings)
        
        # Create figure with 2 rows (Pearson, LERM) x 3 columns (NoS, FA, SIFT2)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        plot_types = ['NoS-weighted', 'FA-weighted', 'SIFT2-weighted']
        # Prettier colors: teal and coral
        colors = {'Traditional': '#26A69A', 'Predicted': '#FF7043'}
        
        for col, ctype in enumerate(plot_types):
            df_ctype = df_plot[df_plot['Connectome Type'] == ctype]
            
            box_pairs=[
                (("84 ROIs", "Traditional"), ("84 ROIs", "Predicted")),
                (("164 ROIs", "Traditional"), ("164 ROIs", "Predicted")),
            ]

            # Pearson's r
            ax = axes[0, col]
            if not df_ctype.empty:
                sns.boxplot(data=df_ctype, x='ROI Count', y='pearson_r', hue='Version',
                           ax=ax, order=['84 ROIs', '164 ROIs'], palette=colors)
                annotator = Annotator(ax, box_pairs, data=df_ctype, x='ROI Count', y='pearson_r', 
                                     hue='Version', order=['84 ROIs', '164 ROIs'])
                annotator.configure(test='Wilcoxon', text_format='star', loc='inside', verbose=0,
                                   pvalue_thresholds=[[1e-3, '***'], [1e-2, '**'], [0.05, '*'], [1, 'ns']])
                annotator.apply_and_annotate()

            ax.set_ylabel("Pearson's r" if col == 0 else "")
            ax.set_xlabel("")
            if ax.get_legend():
                ax.legend_.remove()
            
            # LERM
            ax = axes[1, col]
            if not df_ctype.empty:
                sns.boxplot(data=df_ctype, x='ROI Count', y='lerm_dist', hue='Version',
                           ax=ax, order=['84 ROIs', '164 ROIs'], palette=colors)
                annotator = Annotator(ax, box_pairs, data=df_ctype, x='ROI Count', y='lerm_dist', 
                                     hue='Version', order=['84 ROIs', '164 ROIs'])
                annotator.configure(test='Wilcoxon', text_format='star', loc='inside', verbose=0,
                                   pvalue_thresholds=[[1e-3, '***'], [1e-2, '**'], [0.05, '*'], [1, 'ns']])
                annotator.apply_and_annotate()

            ax.set_ylabel("LERM Distance" if col == 0 else "")
            ax.set_xlabel("")
            if ax.get_legend():
                ax.legend_.remove()

        # Set titles above the top row after annotations are done
        for col, ctype in enumerate(plot_types):
            axes[0, col].set_title(ctype, pad=10)

        # Add legend to bottom right subplot
        handles, labels = axes[0,0].get_legend_handles_labels()
        axes[1, 2].legend(handles, labels, loc='lower right')

        plt.tight_layout()
        
        out_plot = self.plots_dir / "trt_summary_boxplot_with_significance.png"
        plt.savefig(out_plot, dpi=300, bbox_inches='tight')
        plt.close()
        self.log(f"Summary plot with significance saved to {out_plot}")
        
        # Create bar plot with error bars
        self._create_barplot_summary(df)
    
    def _create_barplot_summary(self, df: pd.DataFrame):
        """Create a bar plot summary with mean ± std."""
        # Map labels
        atlas_map = {'aparc+aseg': '84', 'aparc.a2009s+aseg': '164'}
        
        # Compute summary
        summary = df.groupby(['atlas', 'connectome_type', 'version']).agg({
            'pearson_r': ['mean', 'std'],
            'lerm_dist': ['mean', 'std']
        }).reset_index()
        summary.columns = ['atlas', 'connectome_type', 'version', 
                          'r_mean', 'r_std', 'lerm_mean', 'lerm_std']
        summary['atlas_label'] = summary['atlas'].map(atlas_map)
        summary['x_label'] = summary['atlas_label'] + ' ' + summary['connectome_type'].str.upper()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get data for each version
        true_data = summary[summary['version'] == 'true'].sort_values('x_label').reset_index(drop=True)
        pred_data = summary[summary['version'] == 'pred'].sort_values('x_label').reset_index(drop=True)
        
        # Use all unique x_labels from both versions
        all_x_labels = sorted(summary['x_label'].unique())
        x_pos = np.arange(len(all_x_labels))
        width = 0.35
        
        # Create mapping for positioning
        x_label_to_pos = {label: i for i, label in enumerate(all_x_labels)}
        
        # Pearson's r
        ax = axes[0]
        if not true_data.empty:
            true_positions = [x_label_to_pos[label] for label in true_data['x_label']]
            ax.bar(np.array(true_positions) - width/2, true_data['r_mean'], width, 
                  yerr=true_data['r_std'], label='Traditional', color='#2196F3', capsize=3)
        if not pred_data.empty:
            pred_positions = [x_label_to_pos[label] for label in pred_data['x_label']]
            ax.bar(np.array(pred_positions) + width/2, pred_data['r_mean'], width, 
                  yerr=pred_data['r_std'], label='Predicted', color='#FF9800', capsize=3)
        
        ax.set_ylabel("Pearson's r (mean ± std)")
        ax.set_xlabel("Atlas & Connectome Type")
        ax.set_title("Test-Retest Pearson Correlation")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_x_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # LERM
        ax = axes[1]
        if not true_data.empty:
            true_positions = [x_label_to_pos[label] for label in true_data['x_label']]
            ax.bar(np.array(true_positions) - width/2, true_data['lerm_mean'], width, 
                  yerr=true_data['lerm_std'], label='Traditional', color='#2196F3', capsize=3)
        if not pred_data.empty:
            pred_positions = [x_label_to_pos[label] for label in pred_data['x_label']]
            ax.bar(np.array(pred_positions) + width/2, pred_data['lerm_mean'], width, 
                  yerr=pred_data['lerm_std'], label='Predicted', color='#FF9800', capsize=3)
        
        ax.set_ylabel("LERM Distance (mean ± std)")
        ax.set_xlabel("Atlas & Connectome Type")
        ax.set_title("Test-Retest LERM Distance")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_x_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        out_plot = self.plots_dir / "trt_summary_barplot.png"
        plt.savefig(out_plot, dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"Bar plot saved to {out_plot}")


def main():
    parser = argparse.ArgumentParser(description="Compute Test-Retest Connectome Similarities")
    parser.add_argument('--subjects_file', type=str, 
                       default="/media/volume/MV_HCP/subjects_tractography_output_TRT.txt",
                       help="Path to subject list file")
    parser.add_argument('--test_path', type=str,
                       default="/media/volume/MV_HCP/HCP_MRtrix_test",
                       help="Path to test data")
    parser.add_argument('--retest_path', type=str,
                       default="/media/volume/MV_HCP/HCP_MRtrix_retest",
                       help="Path to retest data")
    parser.add_argument('--no_diagonal', action='store_true', 
                       help="Exclude diagonal elements from calculation")
    parser.add_argument('--max_subjects', type=int, default=None,
                       help="Maximum number of subjects to process")
    parser.add_argument('--atlases', nargs='+', 
                       default=["aparc+aseg", "aparc.a2009s+aseg"],
                       help="Atlases to process")
    parser.add_argument('--connectome_types', nargs='+',
                       default=["nos", "fa", "sift2"],
                       help="Connectome types to process")
    
    args = parser.parse_args()
    
    # Check if subjects file exists
    if not os.path.exists(args.subjects_file):
        print(f"ERROR: Subjects file not found: {args.subjects_file}")
        sys.exit(1)
    
    # Run analysis
    analyzer = TRTSimilarityAnalysis(
        subject_list_file=args.subjects_file,
        test_base_path=args.test_path,
        retest_base_path=args.retest_path,
        max_subjects=args.max_subjects,
        no_diagonal=args.no_diagonal,
        atlases=args.atlases,
        connectome_types=args.connectome_types
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
