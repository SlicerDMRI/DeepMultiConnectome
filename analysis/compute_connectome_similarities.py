#!/usr/bin/env python3
"""
Compute Connectome Similarities

This script computes similarity metrics between predicted connectomes and various ground truth references:
1. Intrasubject: Predicted vs True (same subject)
2. Intersubject: Predicted vs True (all other subjects)
3. Population: Predicted vs Training Population Average

Metrics:
- Pearson Correlation (r)
- LERM (Log-Euclidean Riemannian Metric)

Categories:
- Atlases: aparc+aseg (84), aparc.a2009s+aseg (164)
- Weights: NOS, FA, SIFT2

Usage:
    python3 compute_connectome_similarities.py --subject-list /path/to/subjects.txt
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import gc
import json
import argparse
from scipy.spatial.distance import cdist, pdist, squareform
import shutil
from typing import Dict, List, Tuple, Optional, Any
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from scipy.linalg import logm, norm
import scipy.stats as stats
# Add path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
sys.path.append(str(Path("/media/volume/HCP_diffusion_MV/DeepMultiConnectome")))

#from utils.logger import create_logger
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Import shared analysis metrics
from analysis.utils.analysis_metrics import (
    apply_zero_mask,
    compute_correlation,
    compute_lerm
)
from analysis.utils.visualize_connectomes import visualize_subject_connectomes

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class ConnectomeSimilarityAnalysis:
    """
    Computes connectome similarities (Intra, Inter, Population) for specified subjects.
    """
   
    def __init__(self, subject_list_file: str, max_subjects: int = None, mask_zeros: bool = False,
                 no_diagonal: bool = False,
                 atlases: List[str] = None, connectome_types: List[str] = None):
       
        self.subject_list_file = Path(subject_list_file)
        self.max_subjects = max_subjects
        self.mask_zeros = mask_zeros
        self.no_diagonal = no_diagonal
       
        self.base_path = Path("/media/volume/MV_HCP")
        self.population_avg_dir = Path("/media/volume/HCP_diffusion_MV/DeepMultiConnectome/analysis/population_results")
        
        # Setup output directories
        results_dir_name = "similarity_results"
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
       
        self.log(f"Initialized analysis for {len(self.subjects)} subjects")
        self.log(f"Atlases: {self.atlases}")
        self.log(f"Connectome types: {self.connectome_types}")
        self.log(f"Results directory: {self.results_dir}")
        
    def _setup_logger(self):
        """Setup a proper logger that writes to file and stdout"""
        log_file = self.results_dir / f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"
       
        logger = logging.getLogger("ConnectomeSimilarity")
        logger.setLevel(logging.INFO)
        logger.handlers = [] # Clear existing handlers
       
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
       
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
       
        return logger

    def log(self, message):
        """Log message to file and console"""
        self.logger.info(message)

    def _load_subject_list(self) -> List[str]:
        if not self.subject_list_file.exists():
            raise FileNotFoundError(f"Subject list file not found: {self.subject_list_file}")
           
        with open(self.subject_list_file, 'r') as f:
            subjects = [line.strip() for line in f if line.strip()]
           
        if self.max_subjects is not None:
            subjects = subjects[:self.max_subjects]
            self.log(f"Limited to first {self.max_subjects} subjects")
           
        return subjects
    def _get_subject_output_dir(self, subject_id: str, atlas: str) -> Path:
        """Get output directory for connectomes - using population average conventions"""
        return self.base_path / "HCP_MRtrix" / subject_id / "output"

    def _get_pred_output_dir(self, subject_id: str, atlas: str) -> Path:
        """Get output directory for predicted connectomes"""
        return self.base_path / "HCP_MRtrix" / subject_id / "TractCloud"

    def load_population_averages(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load population averages (computed by create_population_connectomes.py)"""
        pop_avgs = {}
        self.log(f"Loading population averages from {self.population_avg_dir}...")
        
        for atlas in self.atlases:
            pop_avgs[atlas] = {}
            for ctype in self.connectome_types:
                # Based on create_population_connectomes.py naming convention
                avg_file = self.population_avg_dir / f"population_average_{atlas}_{ctype}.csv"
                if avg_file.exists():
                    try:
                        # Load raw matrix (no header/index)
                        pop_avgs[atlas][ctype] = pd.read_csv(avg_file, header=None).values
                        self.log(f"  Loaded {atlas} {ctype}")
                    except Exception as e:
                        self.log(f"  Error loading {avg_file}: {e}")
                else:
                    self.log(f"  Missing population average for {atlas} {ctype}")
        return pop_avgs

    def load_subject_connectomes(self, subject_id: str) -> Dict:
        """Load connectomes: pred and true from analysis folder"""
        data = {}
        
        # Helper to get filename
        def get_filename(atlas, ctype, is_true=True):
            mode = "true" if is_true else "pred"
            return f"connectome_{mode}_{ctype}_{atlas}.csv"

        for atlas in self.atlases:
            data[atlas] = {}
            # New path convention: HCP_MRtrix/<id>/analysis/<atlas>
            analysis_dir = self.base_path / "HCP_MRtrix" / subject_id / "analysis" / atlas
           
            for ctype in self.connectome_types:
                t_file = analysis_dir / get_filename(atlas, ctype, is_true=True)
                p_file = analysis_dir / get_filename(atlas, ctype, is_true=False)
                
                # Try to load if files exist
                if t_file.exists() and p_file.exists():
                    try:
                        # Load raw matrices (header=None)
                        t_mat = pd.read_csv(t_file, header=None).values
                        p_mat = pd.read_csv(p_file, header=None).values
                        
                        # Handle NaNs
                        t_mat = np.nan_to_num(t_mat, nan=0.0)
                        p_mat = np.nan_to_num(p_mat, nan=0.0)
                        
                        # Ensure Same Size (just in case)
                        if t_mat.shape != p_mat.shape:
                            # self.log(f"Shape mismatch for {subject_id} {atlas} {ctype}: {t_mat.shape} vs {p_mat.shape}")
                            continue

                        data[atlas][ctype] = {'true': t_mat, 'pred': p_mat}
                    except Exception as e:
                        # self.log(f"Error loading {subject_id} {atlas} {ctype}: {e}")
                        pass
        return data

    def compute_all_comparisons_vectorized(self, all_subjects_data: Dict, population_avgs: Dict) -> List[Dict]:
        """
        Compute all comparisons (Intra, Inter, Group) using vectorized operations.
        Speeds up computation by avoiding nested loops.
        """
        results = []
        subjects = list(all_subjects_data.keys())
        n_subs = len(subjects)
        
        # Pre-compute log matrices and flatten data for all subjects
        # We process one atlas/ctype combination at a time to save memory
        
        # Flatten tasks for progress bar
        tasks = [(a, c) for a in self.atlases for c in self.connectome_types]
        
        for atlas, ctype in tqdm(tasks, desc="Computing Metrics"):
            # Indentation shim to match body
            if True:
                # self.logger.info(f"Processing {atlas} - {ctype}...")
                pass
                
                # 1. Collect Data Arrays
                # ----------------------
                valid_subs = []
                preds = []
                trues = []
                
                # Check dimensions from first valid subject
                dim = None
                for sub in subjects:
                     if atlas in all_subjects_data[sub] and ctype in all_subjects_data[sub][atlas]:
                         dim = all_subjects_data[sub][atlas][ctype]['true'].shape[0]
                         break
                
                if dim is None: continue
                
                # Extract Upper Triangle Indices (including diagonal if requested, but standard is usually without)
                # User instruction 3: "extract only the off-diagonal... values"
                # Standard Pearson for connectomes usually includes or excludes diagonal. 
                # Our metric `compute_correlation` has `include_diagonal`.
                # If include_diagonal=True (k=0), use triu(k=0).
                k = 0 if not self.no_diagonal else 1
                triu_idx = np.triu_indices(dim, k=k)
                
                for sub in subjects:
                    if atlas in all_subjects_data[sub] and ctype in all_subjects_data[sub][atlas]:
                        p = all_subjects_data[sub][atlas][ctype]['pred']
                        t = all_subjects_data[sub][atlas][ctype]['true']
                        
                        # Flatten using upper triangle
                        preds.append(p[triu_idx])
                        trues.append(t[triu_idx])
                        valid_subs.append(sub)
                
                if not valid_subs: continue
                
                preds_flat = np.array(preds) # (N, Features)
                trues_flat = np.array(trues) # (N, Features)
                
                n_valid = len(valid_subs)
                
                # 2. Pearson Correlation 
                # -----------------------------------
                # Intrasubject (Pred_i vs True_i): Row-wise correlation
                # Vectorized Pearson Row-wise:
                # corr(x, y) = ((x - mean_x) . (y - mean_y)) / (std_x * std_y * len)
                # Using cdist is cleaner but standard cdist computes ALL pairs.
                # For Intra, we just need the diagonal of the cross-correlation matrix between Preds and Trues.
                
                # Z-score normalization for Dot Product trick
                def z_score(arr):
                    means = arr.mean(axis=1, keepdims=True)
                    stds = arr.std(axis=1, keepdims=True)
                    return (arr - means) / (stds + 1e-10)
                
                p_z = z_score(preds_flat)
                t_z = z_score(trues_flat)
                
                # Intra: element-wise multiply and sum, divide by N_features
                # Actually, dot product of row vectors = sum(a*b). Correlation is dot(z_a, z_b) / N_features.
                # We can just use the diagonal of the full dot product matrix, or compute row-wise dot manually.
                intra_r_vals = np.sum(p_z * t_z, axis=1) / preds_flat.shape[1]
                
                # Inter: (Pred_i vs True_j where i != j)
                # Compute full cross-correlation matrix (Preds vs Trues)
                # shape (N, N)
                full_corr_mat = np.dot(p_z, t_z.T) / preds_flat.shape[1]
                
                # Mask diagonal to get inter values
                mask_diag = ~np.eye(n_valid, dtype=bool)
                inter_r_means = [np.mean(full_corr_mat[i, mask_diag[i]]) for i in range(n_valid)]

                # Group (Pred vs Pop)
                pop_r_vals = [np.nan] * n_valid
                if atlas in population_avgs and ctype in population_avgs[atlas]:
                    pop_avg_full = population_avgs[atlas][ctype]
                    pop_flat = pop_avg_full[triu_idx].reshape(1, -1)
                    pop_z = z_score(pop_flat)
                    
                    # Dot product between all preds and the single pop avg
                    # (N, F) dot (1, F).T -> (N, 1)
                    pop_r_vals = (np.dot(p_z, pop_z.T) / preds_flat.shape[1]).flatten()

                # 3. LERM 
                # --------------------
                # Needs Logm of full matrices
                # We need to re-loop to compute logms or do it in parallel before.
                # Since we are inside the loop, let's just do it here. 
                # For 164x164, logm is fast enough per subject (order of ms).
                
                p_logs_flat = []
                t_logs_flat = []
                
                # Helper for logm
                def get_logm_flat(mat, dim, epsilon=1e-10):
                    m = mat.copy()
                    np.fill_diagonal(m, m.diagonal() + epsilon)
                    try:
                         # disp=False returns (logm, errest)
                         res = logm(m, disp=False)
                         lm = res[0] if isinstance(res, tuple) else res
                         return np.real(lm).ravel() # Flatten full matrix for LERM expansion trick
                    except:
                         return np.zeros(dim*dim) * np.nan

                # Compute logms
                for i, sub in enumerate(valid_subs):
                     # Retrieve original full matrices again
                     p_full = all_subjects_data[sub][atlas][ctype]['pred']
                     t_full = all_subjects_data[sub][atlas][ctype]['true']
                     
                     p_logs_flat.append(get_logm_flat(p_full, dim))
                     t_logs_flat.append(get_logm_flat(t_full, dim))
                     
                p_logs = np.array(p_logs_flat) # (N, D*D)
                t_logs = np.array(t_logs_flat) # (N, D*D)
                
                # Check for NaNs (failed logm)
                valid_lerm_mask = ~np.isnan(p_logs).any(axis=1) & ~np.isnan(t_logs).any(axis=1)
                
                # Expansion Trick for Euclidean Distance between Flattened Logms
                # ||A - B||^2 = ||A||^2 + ||B||^2 - 2<A, B>
                # A = Pred Logs, B = True Logs
                
                # LERM Intra (Diagonal of distance matrix)
                # dist(A_i, B_i)
                diffs = p_logs - t_logs
                intra_lerm_vals = norm(diffs, axis=1) # Row-wise frobenius norm (since flattened)

                # LERM Inter (Full Distance Matrix)
                # cdist(A, B) uses the same expansion trick internally efficiently
                # Preds vs Trues
                full_lerm_mat = cdist(p_logs, t_logs, metric='euclidean')
                
                inter_lerm_means = [np.mean(full_lerm_mat[i, mask_diag[i]]) for i in range(n_valid)]
                
                # LERM Group
                pop_lerm_vals = [np.nan] * n_valid
                if atlas in population_avgs and ctype in population_avgs[atlas]:
                     pop_full = population_avgs[atlas][ctype]
                     pop_log_flat = get_logm_flat(pop_full, dim).reshape(1, -1)
                     
                     # Distances from all Preds to Pop Log
                     pop_lerm_vals = cdist(p_logs, pop_log_flat, metric='euclidean').flatten()

                # 4. Store Results
                # ----------------
                for i, sub in enumerate(valid_subs):
                    # Intra
                    results.append({
                        'subject_id': sub, 'atlas': atlas, 'connectome_type': ctype,
                        'comparison_type': 'Intrasubject',
                        'pearson_r': intra_r_vals[i], 
                        'lerm_dist': intra_lerm_vals[i] if valid_lerm_mask[i] else np.nan
                    })
                    
                    # Inter
                    results.append({
                        'subject_id': sub, 'atlas': atlas, 'connectome_type': ctype,
                        'comparison_type': 'Intersubject',
                        'pearson_r': inter_r_means[i], 
                        'lerm_dist': inter_lerm_means[i] if valid_lerm_mask[i] else np.nan
                    })
                    
                    # Group
                    results.append({
                        'subject_id': sub, 'atlas': atlas, 'connectome_type': ctype,
                        'comparison_type': 'Group-average',
                        'pearson_r': pop_r_vals[i], 
                        'lerm_dist': pop_lerm_vals[i] if valid_lerm_mask[i] else np.nan
                    })

        return results

    def run_analysis(self):
        """
        Main execution flow.
        """
        self.logger.info("Starting analysis...")
        
        # Example Plots
        if self.subjects:
            first_subject = self.subjects[0]
            self.logger.info(f"Generating example connectome plots for subject {first_subject}...")
            try:
                visualize_subject_connectomes(
                    first_subject, 
                    str(self.base_path), 
                    str(self.plots_dir / "examples")
                )
            except Exception as e:
                self.logger.error(f"Failed to generate example plots: {e}")
        
        # 1. Load Population Averages
        self.logger.info("Loading population averages...")
        population_avgs = self.load_population_averages()

        # 2. Load All Subject Connectomes
        self.logger.info(f"Loading connectomes for {len(self.subjects)} subjects...")
        all_subjects_data = {}
        
        # Parallel loading for speed
        with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
            future_to_sub = {executor.submit(self.load_subject_connectomes, sub): sub for sub in self.subjects}
            
            for future in tqdm(as_completed(future_to_sub), total=len(self.subjects), desc="Loading Data"):
                sub = future_to_sub[future]
                try:
                    data = future.result()
                    if data:
                        all_subjects_data[sub] = data
                except Exception as e:
                    self.logger.error(f"Error loading {sub}: {e}")

        if not all_subjects_data:
            self.logger.error("No subject data loaded. Exiting.")
            return

        self.logger.info(f"Successfully loaded {len(all_subjects_data)} subjects.")

        # 3. Compute Comparisons 
        self.logger.info("Computing similarity metrics (Intra, Inter, Group)...")
        all_results = self.compute_all_comparisons_vectorized(all_subjects_data, population_avgs)

        # 4. Save Results
        if not all_results:
            self.logger.warning("No results generated.")
            return

        df_results = pd.DataFrame(all_results)
        out_csv = self.results_dir / "connectome_similarity_results.csv"
        df_results.to_csv(out_csv, index=False)
        self.logger.info(f"Results saved to {out_csv}")

        # 5. Create Summary Statistics
        self.create_summary_statistics(df_results)

        # 6. Compute Significance Tests
        self.compute_significance_tests(df_results)

        # 7. Create Plot
        self.create_summary_plot(df_results)
        self.logger.info("Analysis Complete.")

    def compute_significance_tests(self, df):
        """
        Compute Wilcoxon signed-rank tests for Intra vs Inter and Intra vs Group.
        Output results to a text file.
        """
        self.logger.info("Computing significance tests...")
        out_file = self.results_dir / "significance_tests.txt"
        
        with open(out_file, 'w') as f:
            f.write("Significance Tests (Wilcoxon Signed-Rank Test)\n")
            f.write("="*60 + "\n\n")
            
            # Group keys
            atlases = df['atlas'].unique()
            ctypes = df['connectome_type'].unique()
            metrics = ['pearson_r', 'lerm_dist']
            
            for atlas in sorted(atlases):
                for ctype in sorted(ctypes):
                    f.write(f"Configuration: Atlas={atlas}, Type={ctype}\n")
                    f.write("-" * 50 + "\n")
                    
                    subset = df[(df['atlas'] == atlas) & (df['connectome_type'] == ctype)]
                    
                    # Prepare data: Pivot to have subjects as index and comparison_type as columns
                    # We need to ensure we have paired data
                    pivot = subset.pivot(index='subject_id', columns='comparison_type', values=metrics)
                    
                    # Flatten columns (metric, comparison_type)
                    # columns will be MultiIndex
                    
                    for metric in metrics:
                        f.write(f"  Metric: {metric}\n")
                        
                        # Extract paired arrays
                        try:
                            # Columns are (metric, type)
                            intra = pivot[(metric, 'Intrasubject')]
                            inter = pivot[(metric, 'Intersubject')]
                            group = pivot[(metric, 'Group-average')]
                            
                            # Drop NaNs for paired tests
                            # Pair: Intra vs Inter
                            pair1 = pd.concat([intra, inter], axis=1).dropna()
                            if len(pair1) > 1:
                                stat, p = stats.wilcoxon(pair1.iloc[:, 0], pair1.iloc[:, 1])
                                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                                f.write(f"    Intra vs Inter:   W={stat:.1f}, p={p:.2e} ({sig})\n")
                            else:
                                f.write("    Intra vs Inter:   Not enough data\n")
                                
                            # Pair: Intra vs Group
                            pair2 = pd.concat([intra, group], axis=1).dropna()
                            if len(pair2) > 1:
                                stat, p = stats.wilcoxon(pair2.iloc[:, 0], pair2.iloc[:, 1])
                                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                                f.write(f"    Intra vs Group:   W={stat:.1f}, p={p:.2e} ({sig})\n")
                            else:
                                f.write("    Intra vs Group:   Not enough data\n")
                                
                        except KeyError as e:
                             f.write(f"    Skipping (Missing Data): {e}\n")
                        
                        f.write("\n")
                    f.write("\n")
        
        self.logger.info(f"Significance tests saved to {out_file}")

    def create_summary_statistics(self, df):
        """
        Generate summary statistics (mean +/- std) for each category.
        """
        self.logger.info("Generating summary statistics...")
        
        # Group by Atlas, Connectome Type, and Comparison Type
        # Calculate mean and std for metrics
        summary = df.groupby(['atlas', 'connectome_type', 'comparison_type']).agg({
            'pearson_r': ['mean', 'std', 'count'],
            'lerm_dist': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        out_summary = self.results_dir / "summary_statistics.csv"
        summary.to_csv(out_summary, index=False)
        self.logger.info(f"Summary statistics saved to {out_summary}")

    def create_summary_plot(self, df):
        """
        Creates a 2x3 subplot figure with Deja Vu font.
        Rows: Pearson's r, LERM
        Cols: NOS, FA, SIFT2
        X-axis: Atlas (84 vs 164)
        Hue: Comparison Type (Intra, Inter, Group)
        Colors: Blue, Red, Green
        """
        self.logger.info("Generating summary plots...")
        
        # Filter for relevant connectome types
        plot_types = ['nos', 'fa', 'sift2']
        df_plot = df[df['connectome_type'].isin(plot_types)].copy()
        
        # Rename atlases for display
        atlas_map = {
            'aparc+aseg': '84 ROIs',
            'aparc.a2009s+aseg': '164 ROIs'
        }
        df_plot['ROI Count'] = df_plot['atlas'].map(atlas_map)
        
        # Setup plot style
        # Force font in seaborn context and style to ensure it sticks
        font_settings = {'font.family': 'DejaVu Sans', 'font.sans-serif': ['DejaVu Sans']}
        sns.set_context("notebook", font_scale=1.2, rc=font_settings)
        sns.set_style("white", rc=font_settings)
        
        # Global matplotlib update just in case
        plt.rcParams.update(font_settings)
        matplotlib.rcParams.update(font_settings)
        
        # Explicit font dictionary for overrides
        fdict = {'family': 'DejaVu Sans'}
        
        # Setup figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
        
        metrics = [('pearson_r', "Pearson's r"), ('lerm_dist', 'LERM Distance')]
        
        # Custom palette
        palette = {
            'Intrasubject': 'royalblue',
            'Intersubject': 'firebrick',
            'Group-average': 'forestgreen'
        }
        
        for row_idx, (metric_col, metric_name) in enumerate(metrics):
            for col_idx, ctype in enumerate(plot_types):
                ax = axes[row_idx, col_idx]
                
                # Subset data
                subset = df_plot[df_plot['connectome_type'] == ctype].copy()
                subset = subset.dropna(subset=[metric_col])

                if subset.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontdict=fdict)
                    continue

                # Boxplot
                sns.boxplot(
                    data=subset,
                    x='ROI Count',
                    y=metric_col,
                    hue='comparison_type',
                    ax=ax,
                    palette=palette,
                    order=['84 ROIs', '164 ROIs'],
                    hue_order=['Intrasubject', 'Intersubject', 'Group-average'],
                    showfliers=False
                )
                
                # Update Headers: Only on top row
                type_labels = {
                    'nos': 'NoS-weighted',
                    'fa': 'FA-weighted',
                    'sift2': 'SIFT2-weighted'
                }
                
                if row_idx == 0:
                    ax.set_title(type_labels[ctype], fontsize=18, fontdict=fdict)
                else:
                    ax.set_title("")
                
                # Labels
                ax.set_xlabel("", fontdict=fdict)
                ax.set_ylabel(metric_name if col_idx == 0 else "", fontdict=fdict)
                
                # Gridlines
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Force Tick Fonts
                ax.tick_params(labelsize=14) 
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname('DejaVu Sans')

                # Remove Legend from all plots
                if ax.get_legend():
                    ax.get_legend().remove()

        # Custom Legend on bottom right subplot (row 1, col 2)
        handles, labels = axes[1, 2].get_legend_handles_labels()
        # If handles are empty (e.g. no data in last plot), grab from first
        if not handles:
             handles, labels = axes[0, 0].get_legend_handles_labels()
             
        if handles:
            axes[1, 2].legend(handles, labels, loc='lower right', frameon=True, prop={'family': 'DejaVu Sans', 'size': 16})

        out_plot = self.plots_dir / "connectome_similarity_summary.png"
        plt.savefig(out_plot, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to {out_plot}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute Connectome Similarities (Intra, Inter, Group)")
    parser.add_argument('--subjects_file', type=str, default="/media/volume/MV_HCP/subjects_tractography_output_1000_test.txt",
                      help="Path to subject list file")
    parser.add_argument('--no_diagonal', action='store_true', help="Exclude diagonal elements from calculation")
    
    # We allow the user to point to a file, or if the file doesn't exist, maybe it's a test run?
    # The user mentions subjects_tractography_output_1000_test.txt in context
    
    args = parser.parse_args()
    
    subjects_file = args.subjects_file
    
    # Check if file exists, if not try to use a default or fail
    if not os.path.exists(subjects_file):
        # Look for a default in the workspace if arg is just a name
        possible_path = Path("/media/volume/MV_HCP") / subjects_file
        if possible_path.exists():
            subjects_file = str(possible_path)
        else:
             print(f"Subject file {subjects_file} not found.")
             # For safety/testing, if arg looks like a subject ID, create a dummy file or list
             if subjects_file.isdigit():
                 print("Treating argument as subject ID.")
                 # But the class takes a file path. 
                 # To keep it simple, we require a file.
                 return

    analyzer = ConnectomeSimilarityAnalysis(
        subject_list_file=subjects_file,
        no_diagonal=args.no_diagonal
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
