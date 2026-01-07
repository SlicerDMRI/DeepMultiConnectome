#!/usr/bin/env python3
"""
Population Average Connectome Analysis

This script compares test subject predictions against population-weighted average 
connectomes computed from the training set.

Analysis:
- Computes population average connectomes from training subjects (true connectomes)
- Compares test subject predictions against these population averages
- Uses same metrics as intra/inter-subject analysis for consistency

Metrics computed using shared analysis_metrics module:
- Pearson correlation (k=0, includes diagonal)
- LERM (Log-Euclidean Riemannian Metric) using matrix logarithm
- Zero masking for diffusion metrics (FA, MD, AD, RD)

Usage:
    python3 population_average_analysis.py --compute-average
    python3 population_average_analysis.py --test-comparison
    python3 population_average_analysis.py --full-analysis
"""

import os
import sys
import argparse
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Add path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from utils.unified_connectome import ConnectomeAnalyzer
from utils.connectome_utils import ConnectomeBuilder
from utils.logger import create_logger

# Import shared analysis metrics
from analysis.utils.analysis_metrics import (
    apply_zero_mask,
    compute_correlation,
    compute_lerm,
    compute_connectome_metrics
)


class PopulationConnectomeAnalysis:
    """
    Population-level connectome analysis combining training set averages with test set validation
    """
    
    def __init__(self, base_path: str = "/media/volume/MV_HCP", out_path: str = '/media/volume/HCP_diffusion_MV/DeepMultiConnectome/analysis'):
        """Initialize the population analysis"""
        
        self.base_path = Path(base_path)

        # Subject lists
        self.train_subjects_file = self.base_path / "subjects_tractography_output_1000_train_200.txt"
        self.test_subjects_file = self.base_path / "subjects_tractography_output_1000_test.txt"
        
        # Output directory
        self.output_dir = Path(out_path) / "population_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Atlas configuration
        self.atlases = ["aparc+aseg", "aparc.a2009s+aseg"]
        self.connectome_types = ["nos", "fa", "sift2"]
        
        # Create logger
        self.logger = create_logger(str(self.output_dir))
        
        # Load subject lists
        self.train_subjects = self._load_subject_list(self.train_subjects_file)
        self.test_subjects = self._load_subject_list(self.test_subjects_file)
        
        print(f"Initialized population analysis")
        print(f"Training subjects: {len(self.train_subjects)}")
        print(f"Test subjects: {len(self.test_subjects)}")
        print(f"Output directory: {self.output_dir}")
        
        self.logger.info(f"Initialized population analysis")
        self.logger.info(f"Training subjects: {len(self.train_subjects)}")
        self.logger.info(f"Test subjects: {len(self.test_subjects)}")
    
    def _load_subject_list(self, subjects_file: Path) -> List[str]:
        """Load subject IDs from file"""
        try:
            with open(subjects_file, 'r') as f:
                subjects = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(subjects)} subjects from {subjects_file}")
            return subjects
        except Exception as e:
            self.logger.error(f"Error loading subjects from {subjects_file}: {e}")
            return []
    
    def _get_subject_paths(self, subject_id: str) -> Dict[str, Path]:
        """Get all relevant paths for a subject"""
        subject_path = self.base_path / "HCP_MRtrix" / subject_id
        
        return {
            'subject_path': subject_path,
            'true_base': subject_path / "output",
            'pred_base': subject_path / "TractCloud",
            'diffusion_dir': subject_path / "dMRI"
        }
    
    def _load_connectome_csv(self, subject_id: str, atlas: str, metric: str = "nos", mode: str = "pred") -> Optional[np.ndarray]:
        """
        Load connectome from CSV file (much faster than labels approach)
        
        Args:
            subject_id: Subject identifier
            atlas: Atlas name ('aparc+aseg' or 'aparc.a2009s+aseg')
            metric: Connectome metric ('nos', 'fa', 'sift2')
            mode: Either 'pred' or 'true'
            
        Returns:
            Connectome matrix or None if loading failed
        """
        # Determine filename based on metric
        if metric == 'nos':
            filename = f"{subject_id}_connectome_{atlas}_{mode}.csv"
        else:
            filename = f"{subject_id}_connectome_{atlas}_{metric}_{mode}.csv"
            
        # Determine the correct directory based on subject type
        # Try multiple possible locations
        possible_paths = [
            os.path.join(self.base_path, "connectomes_test", filename),
            os.path.join(self.base_path, "connectomes_200", filename),
            os.path.join(self.base_path, f"connectomes_{mode}", filename)
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
        
        if csv_path is None:
            return None
        
        try:
            connectome = pd.read_csv(csv_path, index_col=0).values
            
            # Handle NaNs
            connectome = np.nan_to_num(connectome, nan=0.0)
            
            # Handle dimension alignment (CSV files may be 83x83 or 163x163)
            expected_size = 84 if atlas == 'aparc+aseg' else 164
            
            if connectome.shape[0] == expected_size - 1:
                # Pad connectome assuming region 0 is missing
                padded_connectome = np.zeros((expected_size, expected_size))
                padded_connectome[1:, 1:] = connectome
                connectome = padded_connectome
                self.logger.debug(f"Padded {mode} connectome from {connectome.shape[0]-1} to {connectome.shape[0]} for {atlas}")
            
            return connectome
            
        except Exception as e:
            self.logger.warning(f"Error loading CSV {csv_path}: {str(e)}")
            return None
    
    def _load_subject_connectome(self, subject_id: str, atlas: str,
                                metric: str = "nos", connectome_type: str = "true") -> Optional[np.ndarray]:
        """
        Load a connectome for a specific subject and atlas
        
        Args:
            subject_id: Subject identifier
            atlas: Atlas name
            metric: Connectome metric ('nos', 'fa', 'sift2')
            connectome_type: 'true' for ground truth, 'pred' for predicted
        
        Returns:
            Connectome matrix or None if loading failed
        """
        paths = self._get_subject_paths(subject_id)
        
        try:
            # Determine filename based on metric
            if connectome_type == "true":
                # Filename conventions for HCP_MRtrix output
                if metric == 'nos':
                    filename = f"connectome_matrix_{atlas}.csv"
                elif metric == 'sift2':
                    filename = f"connectome_matrix_SIFT_sum_{atlas}.csv"
                elif metric in ['fa', 'md', 'ad', 'rd']:
                    filename = f"connectome_matrix_{metric.upper()}_mean_{atlas}.csv"
                else:
                    filename = f"connectome_matrix_{atlas}_{metric}.csv"
            else:
                # Filename conventions for predicted output
                if metric == 'nos':
                    filename = f"connectome_matrix_{atlas}.csv"
                else:
                    filename = f"connectome_matrix_{atlas}_{metric}.csv"

            if connectome_type == "true":
                # Try to load pre-computed connectome matrix first
                connectome_file = paths['true_base'] / filename
                
                if connectome_file.exists():
                    # Load pre-computed connectome matrix
                    connectome_df = pd.read_csv(connectome_file, index_col=0)
                    connectome = connectome_df.values
                    
                    # Handle NaNs (common in diffusion metrics where no streamlines exist)
                    connectome = np.nan_to_num(connectome, nan=0.0)
                    
                    # Ensure proper dimensions - pad if necessary to match expected atlas dimensions
                    expected_dims = {'aparc+aseg': 84, 'aparc.a2009s+aseg': 164}
                    
                    if atlas in expected_dims:
                        expected_dim = expected_dims[atlas]
                        if connectome.shape[0] == expected_dim - 1:
                            # Pad with zeros to match expected dimensions (add row/column for missing region)
                            padded_connectome = np.zeros((expected_dim, expected_dim))
                            padded_connectome[1:, 1:] = connectome  # Assume region 0 is missing
                            connectome = padded_connectome
                            self.logger.debug(f"Padded connectome from {connectome_df.shape} to {connectome.shape}")
                    
                    self.logger.debug(f"Successfully loaded pre-computed {connectome_type} connectome for {subject_id} {atlas}")
                    return connectome
                else:
                    # Fallback to labels file approach (Only for NOS)
                    if metric == 'nos':
                        labels_file = paths['true_base'] / f"labels_100K_{atlas}_symmetric.txt"
                        if not labels_file.exists():
                            # Try alternative naming
                            labels_file = paths['true_base'] / f"labels_10M_{atlas}_symmetric.txt"
                        
                        if labels_file.exists():
                            # Use ConnectomeAnalyzer to create connectome
                            analyzer = ConnectomeAnalyzer(atlas=atlas, out_path=str(self.output_dir), logger=self.logger)
                            
                            # Load labels
                            with open(labels_file, 'r') as f:
                                labels = [int(line.strip()) for line in f if line.strip()]
                            
                            # Create NOS (Number of Streamlines) connectome
                            connectome = analyzer.create_connectome_from_labels(labels, connectome_name='nos')
                            
                            if connectome is not None:
                                self.logger.debug(f"Successfully created {connectome_type} connectome from labels for {subject_id} {atlas}")
                                return connectome
            else:
                # For predicted connectomes, try TractCloud directory first
                connectome_file = paths['pred_base'] / filename
                
                if connectome_file.exists():
                    # Load pre-computed predicted connectome matrix
                    connectome_df = pd.read_csv(connectome_file, index_col=0)
                    connectome = connectome_df.values
                    
                    # Handle NaNs
                    connectome = np.nan_to_num(connectome, nan=0.0)
                    
                    # Ensure proper dimensions - pad if necessary to match expected atlas dimensions
                    expected_dims = {'aparc+aseg': 84, 'aparc.a2009s+aseg': 164}
                    
                    if atlas in expected_dims:
                        expected_dim = expected_dims[atlas]
                        if connectome.shape[0] == expected_dim - 1:
                            # Pad with zeros to match expected dimensions (add row/column for missing region)
                            padded_connectome = np.zeros((expected_dim, expected_dim))
                            padded_connectome[1:, 1:] = connectome  # Assume region 0 is missing
                            connectome = padded_connectome
                            self.logger.debug(f"Padded connectome from {connectome_df.shape} to {connectome.shape}")
                    
                    self.logger.debug(f"Successfully loaded pre-computed {connectome_type} connectome for {subject_id} {atlas}")
                    return connectome
                else:
                    # Fallback to predictions file approach
                    if metric == 'nos':
                        labels_file = paths['pred_base'] / f"predictions_{atlas}_symmetric.txt"
                        
                        if labels_file.exists():
                            # Use ConnectomeAnalyzer to create connectome
                            analyzer = ConnectomeAnalyzer(atlas=atlas, out_path=str(self.output_dir), logger=self.logger)
                            
                            # Load labels
                            with open(labels_file, 'r') as f:
                                labels = [int(line.strip()) for line in f if line.strip()]
                            
                            # Create NOS (Number of Streamlines) connectome
                            connectome = analyzer.create_connectome_from_labels(labels, connectome_name='nos')
                            
                            if connectome is not None:
                                self.logger.debug(f"Successfully created {connectome_type} connectome from labels for {subject_id} {atlas}")
                                return connectome
            
            self.logger.warning(f"No connectome data found for {subject_id} {atlas} ({connectome_type})")
            return None
                
        except Exception as e:
            self.logger.error(f"Error loading connectome for {subject_id} {atlas}: {e}")
            return None
    
    def compute_population_average_connectomes(self, force_recompute: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute population average connectomes from training subjects
        
        Args:
            force_recompute: Whether to recompute even if cached files exist
            
        Returns:
            Dictionary with atlas names as keys and a sub-dictionary of connectome types and average matrices
        """
        print(f"\n{'='*80}")
        print("COMPUTING POPULATION AVERAGE CONNECTOMES")
        print(f"{'='*80}")
        self.logger.info("Computing population average connectomes")
        
        average_connectomes = {}
        
        for atlas in self.atlases:
            print(f"\nProcessing atlas: {atlas}")
            self.logger.info(f"Processing atlas: {atlas}")
            average_connectomes[atlas] = {}
            
            for connectome_type in self.connectome_types:
                print(f"  Processing connectome type: {connectome_type.upper()}")
                
                # Check for cached file
                cache_file_npy = self.output_dir / f"population_average_{atlas}_{connectome_type}.npy"
                cache_file_csv = self.output_dir / f"population_average_{atlas}_{connectome_type}.csv"
                
                # Also check legacy cache filename for NOS
                legacy_cache_file = self.output_dir / f"population_average_{atlas}.npy"
                
                if cache_file_npy.exists() and not force_recompute:
                    print(f"    Loading cached average connectome from {cache_file_npy.name}")
                    try:
                        average_connectomes[atlas][connectome_type] = np.load(cache_file_npy)
                        self.logger.info(f"Loaded cached average connectome for {atlas} {connectome_type}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached file {cache_file_npy}: {e}")
                elif connectome_type == 'nos' and legacy_cache_file.exists() and not force_recompute:
                     print(f"    Loading legacy cached average connectome from {legacy_cache_file.name}")
                     try:
                        average_connectomes[atlas][connectome_type] = np.load(legacy_cache_file)
                        self.logger.info(f"Loaded legacy cached average connectome for {atlas}")
                        continue
                     except Exception as e:
                        self.logger.warning(f"Failed to load legacy cached file {legacy_cache_file}: {e}")

                
                # Compute average from training subjects
                print(f"    Computing average from {len(self.train_subjects)} training subjects...")
                
                connectomes = []
                successful_subjects = []
                
                for i, subject_id in enumerate(self.train_subjects):
                    if i % 100 == 0:  # Progress update every 100 subjects
                        print(f"      Progress: {i}/{len(self.train_subjects)} subjects processed")
                    
                    connectome = self._load_subject_connectome(subject_id, atlas, metric=connectome_type, connectome_type="true")
                    
                    if connectome is not None:
                        # Append directly, NaNs are already handled in _load_subject_connectome
                        connectomes.append(connectome)
                        successful_subjects.append(subject_id)
                
                print(f"    Successfully loaded {len(connectomes)} connectomes from {len(self.train_subjects)} subjects")
                self.logger.info(f"Successfully loaded {len(connectomes)} connectomes for {atlas} {connectome_type}")
                
                if len(connectomes) == 0:
                    self.logger.warning(f"No connectomes loaded for {atlas} {connectome_type}")
                    continue
                
                # Compute average
                print(f"    Computing population average...")
                connectomes_array = np.array(connectomes)
                average_connectome = np.mean(connectomes_array, axis=0)
                
                # Save average connectome as NPY
                np.save(cache_file_npy, average_connectome)
                print(f"    Saved average connectome to {cache_file_npy.name}")

                # Save average connectome as CSV
                pd.DataFrame(average_connectome).to_csv(cache_file_csv)
                print(f"    Saved average connectome to {cache_file_csv.name}")
                
                average_connectomes[atlas][connectome_type] = average_connectome
                
                # Print statistics
                print(f"    Average connectome computed")
                print(f"      Shape: {average_connectome.shape}")
                print(f"      Non-zero connections: {np.count_nonzero(average_connectome)}")
                print(f"      Mean value: {np.mean(average_connectome):.2f}")
                print(f"      Max value: {np.max(average_connectome):.2f}")
                
                # Save list of successful subjects
                subjects_file = self.output_dir / f"successful_train_subjects_{atlas}_{connectome_type}.txt"
                with open(subjects_file, 'w') as f:
                    for subject in successful_subjects:
                        f.write(f"{subject}\n")
                
                self.logger.info(f"Average connectome computed for {atlas} {connectome_type}")
                self.logger.info(f"Shape: {average_connectome.shape}")
                self.logger.info(f"Non-zero connections: {np.count_nonzero(average_connectome)}")
        
        return average_connectomes
    
    def compare_test_subjects_to_population(self, average_connectomes: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
        """
        Compare all test subjects' predicted connectomes to population averages for all connectome types
        
        Args:
            average_connectomes: Nested dictionary {atlas: {connectome_type: average_matrix}}
            
        Returns:
            DataFrame with comparison results for all test subjects and connectome types
        """
        print(f"\n{'='*80}")
        print("COMPARING TEST SUBJECTS TO POPULATION AVERAGES")
        print(f"{'='*80}")
        self.logger.info("Comparing test subjects to population averages")
        
        results = []
        
        for atlas in self.atlases:
            if atlas not in average_connectomes:
                self.logger.warning(f"No average connectomes available for {atlas}")
                continue
            
            print(f"\nProcessing atlas: {atlas}")
            
            for connectome_type in self.connectome_types:
                if connectome_type not in average_connectomes[atlas]:
                    print(f"  Skipping {connectome_type.upper()} (no population average available)")
                    continue
                
                print(f"\n  Connectome type: {connectome_type.upper()}")
                average_connectome = average_connectomes[atlas][connectome_type]
                
                successful_comparisons = 0
                
                for i, subject_id in enumerate(self.test_subjects):
                    if i % 50 == 0 and i > 0:  # Progress update every 50 subjects
                        print(f"      Progress: {i}/{len(self.test_subjects)} subjects processed")
                    
                    # Load predicted connectome
                    pred_connectome = self._load_connectome_csv(subject_id, atlas, connectome_type, "pred")
                    
                    if pred_connectome is None and connectome_type == 'nos':
                        # Fallback to labels method (only for NOS)
                        pred_connectome = self._load_subject_connectome(subject_id, atlas, metric=connectome_type, connectome_type="pred")
                    
                    if pred_connectome is None:
                        continue
                    
                    # Load true connectome for additional comparison
                    true_connectome = self._load_connectome_csv(subject_id, atlas, connectome_type, "true")
                    
                    if true_connectome is None and connectome_type == 'nos':
                        # Fallback to labels method (only for NOS)
                        true_connectome = self._load_subject_connectome(subject_id, atlas, metric=connectome_type, connectome_type="true")
                    
                    # Compare predicted to population average
                    pred_vs_pop = self._compute_comparison_metrics(
                        pred_connectome, average_connectome, connectome_type=connectome_type
                    )
                    
                    # Compare true to population average (for reference)
                    true_vs_pop = None
                    if true_connectome is not None:
                        true_vs_pop = self._compute_comparison_metrics(
                            true_connectome, average_connectome, connectome_type=connectome_type
                        )
                    
                    # Compare predicted to true (individual subject accuracy)
                    pred_vs_true = None
                    if true_connectome is not None:
                        pred_vs_true = self._compute_comparison_metrics(
                            pred_connectome, true_connectome, connectome_type=connectome_type
                        )
                    
                    # Store results
                    result = {
                        'subject_id': subject_id,
                        'atlas': atlas,
                        'connectome_type': connectome_type,
                        'pred_vs_pop_correlation': pred_vs_pop['correlation'],
                        'pred_vs_pop_rmse': pred_vs_pop['rmse'],
                        'pred_vs_pop_lerm': pred_vs_pop['lerm'],
                        'pred_vs_pop_mae': pred_vs_pop['mae'],
                        'pred_connections': pred_vs_pop['pred_connections'],
                        'pop_connections': pred_vs_pop['true_connections']
                    }
                    
                    if true_vs_pop is not None:
                        result.update({
                            'true_vs_pop_correlation': true_vs_pop['correlation'],
                            'true_vs_pop_rmse': true_vs_pop['rmse'],
                            'true_vs_pop_lerm': true_vs_pop['lerm']
                        })
                    
                    if pred_vs_true is not None:
                        result.update({
                            'pred_vs_true_correlation': pred_vs_true['correlation'],
                            'pred_vs_true_rmse': pred_vs_true['rmse'],
                            'pred_vs_true_lerm': pred_vs_true['lerm']
                        })
                    
                    results.append(result)
                    successful_comparisons += 1
                
                print(f"      Completed {successful_comparisons} comparisons for {connectome_type.upper()}")
                self.logger.info(f"Completed {successful_comparisons} comparisons for {atlas} {connectome_type}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Save results
            results_file = self.output_dir / "test_vs_population_results.csv"
            results_df.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")
            self.logger.info(f"Results saved to: {results_file}")
        else:
            print("\nNo results to save (no connectomes were successfully compared)")
            self.logger.warning("No results to save")
        
        return results_df
    
    def _compute_comparison_metrics(self, pred_connectome: np.ndarray, 
                                  population_avg_connectome: np.ndarray,
                                  connectome_type: str = 'nos') -> Dict[str, float]:
        """
        Compute comparison metrics between prediction and population average
        
        Uses shared metrics module to ensure consistency with intra/inter-subject analysis.
        
        Args:
            pred_connectome: Predicted connectome matrix
            population_avg_connectome: Population average connectome matrix
            connectome_type: Type of connectome (nos, fa, md, ad, rd, sift2)
            
        Returns:
            Dictionary with comparison metrics
        """
        # Check and align dimensions
        if pred_connectome.shape != population_avg_connectome.shape:
            self.logger.warning(f"Dimension mismatch: pred {pred_connectome.shape} vs avg {population_avg_connectome.shape}")
            
            # Find the target dimension (larger one, or expected dimension)
            max_dim = max(pred_connectome.shape[0], population_avg_connectome.shape[0])
            
            # Pad smaller connectome to match larger one
            if pred_connectome.shape[0] < max_dim:
                padded_pred = np.zeros((max_dim, max_dim))
                padded_pred[:pred_connectome.shape[0], :pred_connectome.shape[1]] = pred_connectome
                pred_connectome = padded_pred
                self.logger.info(f"Padded predicted connectome to {pred_connectome.shape}")
            
            if population_avg_connectome.shape[0] < max_dim:
                padded_avg = np.zeros((max_dim, max_dim))
                padded_avg[:population_avg_connectome.shape[0], :population_avg_connectome.shape[1]] = population_avg_connectome
                population_avg_connectome = padded_avg
                self.logger.info(f"Padded population average connectome to {population_avg_connectome.shape}")
            
            self.logger.info(f"Aligned connectomes to {pred_connectome.shape}")
        
        # Use shared metrics computation for consistency with intra/inter-subject analysis
        # Note: We compare prediction to population average, treating it like a "true" reference
        metrics = compute_connectome_metrics(
            true_matrix=population_avg_connectome,  # Population average as reference
            pred_matrix=pred_connectome,
            connectome_type=connectome_type,
            mask_zeros=False,  # Apply zero masking for diffusion metrics
            logger=self.logger
        )
        
        # Also compute RMSE and MAE for backwards compatibility
        mask = ~np.eye(pred_connectome.shape[0], dtype=bool)
        pred_flat = pred_connectome[mask].flatten()
        avg_flat = population_avg_connectome[mask].flatten()
        
        # For diffusion metrics, apply zero masking
        # if connectome_type in ['fa', 'md', 'ad', 'rd']:
        #    nonzero_mask = (pred_flat != 0) & (avg_flat != 0)
        #    pred_flat = pred_flat[nonzero_mask]
        #    avg_flat = avg_flat[nonzero_mask]
        
        rmse = np.sqrt(mean_squared_error(avg_flat, pred_flat))
        mae = np.mean(np.abs(pred_flat - avg_flat))
        
        return {
            'correlation': metrics['correlation'],
            'lerm': metrics['lerm'],
            'rmse': rmse,
            'mae': mae,
            'pred_connections': metrics['pred_connections'],
            'true_connections': metrics['true_connections']  # This is the population avg connections
        }
    
    def create_population_analysis_report(self, results_df: pd.DataFrame, 
                                        average_connectomes: Dict[str, np.ndarray]):
        """
        Create comprehensive population analysis report with visualizations
        
        Args:
            results_df: DataFrame with comparison results
            average_connectomes: Dictionary of population average connectomes
        """
        print(f"\n{'='*80}")
        print("CREATING POPULATION ANALYSIS REPORT")
        print(f"{'='*80}")
        self.logger.info("Creating population analysis report")
        
        # Create summary statistics
        self._create_summary_statistics(results_df)
        
        # Create comparison plots
        self._create_comparison_plots(results_df)
        
        # Create population connectome visualizations using unified_connectome
        self._create_population_connectome_plots(average_connectomes)
        
        # Create connectome visualizations (simple heatmaps)
        self._create_connectome_visualizations(average_connectomes)
        
        # Create individual subject comparison plots (5 subjects)
        self._create_individual_comparison_plots(results_df, average_connectomes)
        
        # Create detailed subject-wise report
        self._create_subject_wise_report(results_df)
        
        # Save metrics as JSON
        self._save_metrics_json(results_df, average_connectomes)
        
        print("Population analysis report completed")
        self.logger.info("Population analysis report completed")
    
    def _create_summary_statistics(self, results_df: pd.DataFrame):
        """Create summary statistics for the population analysis"""
        print("\nCreating summary statistics...")
        
        if len(results_df) == 0:
            print("  No results to summarize")
            return
        
        summary_stats = {}
        
        # Group by atlas and connectome_type
        for atlas in self.atlases:
            for connectome_type in self.connectome_types:
                # Filter data for this combination
                mask = (results_df['atlas'] == atlas) & (results_df['connectome_type'] == connectome_type)
                data = results_df[mask]
                
                if len(data) == 0:
                    continue
                
                key = f"{atlas}_{connectome_type}"
                
                stats = {
                    'atlas': atlas,
                    'connectome_type': connectome_type,
                    'n_subjects': len(data),
                    'pred_vs_pop_correlation_mean': data['pred_vs_pop_correlation'].mean(),
                    'pred_vs_pop_correlation_std': data['pred_vs_pop_correlation'].std(),
                    'pred_vs_pop_rmse_mean': data['pred_vs_pop_rmse'].mean(),
                    'pred_vs_pop_rmse_std': data['pred_vs_pop_rmse'].std(),
                    'pred_vs_pop_lerm_mean': data['pred_vs_pop_lerm'].mean(),
                    'pred_vs_pop_lerm_std': data['pred_vs_pop_lerm'].std()
                }
                
                # Add true vs population statistics (natural variability baseline)
                if 'true_vs_pop_correlation' in data.columns:
                    stats.update({
                        'true_vs_pop_correlation_mean': data['true_vs_pop_correlation'].mean(),
                        'true_vs_pop_correlation_std': data['true_vs_pop_correlation'].std(),
                        'true_vs_pop_rmse_mean': data['true_vs_pop_rmse'].mean(),
                        'true_vs_pop_rmse_std': data['true_vs_pop_rmse'].std(),
                        'true_vs_pop_lerm_mean': data['true_vs_pop_lerm'].mean(),
                        'true_vs_pop_lerm_std': data['true_vs_pop_lerm'].std()
                    })
                
                # Add individual accuracy if available
                if 'pred_vs_true_correlation' in data.columns:
                    stats.update({
                        'pred_vs_true_correlation_mean': data['pred_vs_true_correlation'].mean(),
                        'pred_vs_true_correlation_std': data['pred_vs_true_correlation'].std(),
                        'pred_vs_true_rmse_mean': data['pred_vs_true_rmse'].mean(),
                        'pred_vs_true_rmse_std': data['pred_vs_true_rmse'].std(),
                        'pred_vs_true_lerm_mean': data['pred_vs_true_lerm'].mean(),
                        'pred_vs_true_lerm_std': data['pred_vs_true_lerm'].std()
                    })
                
                summary_stats[key] = stats
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats).T
        summary_file = self.output_dir / "population_summary_statistics.csv"
        summary_df.to_csv(summary_file)
        
        # Print summary (grouped by connectome type for readability)
        print("\nPopulation Summary Statistics:")
        for connectome_type in self.connectome_types:
            type_data = summary_df[summary_df['connectome_type'] == connectome_type]
            if len(type_data) > 0:
                print(f"\n{connectome_type.upper()}:")
                print(type_data[['atlas', 'n_subjects', 'pred_vs_pop_correlation_mean', 
                               'pred_vs_pop_lerm_mean']].round(4).to_string(index=False))
        
        self.logger.info("Summary statistics created")
    
    def _create_comparison_plots(self, results_df: pd.DataFrame):
        """Create comparison plots for population analysis"""
        print("\nCreating comparison plots...")
        
        if len(results_df) == 0:
            print("  No results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Population Connectome Analysis Results', fontsize=16, fontweight='bold')
        
        # Helper to plot boxplot safely
        def safe_boxplot(data, x, y, hue, ax, title, ylabel):
            # Drop NaNs for the y-variable to avoid seaborn errors
            plot_data = data.dropna(subset=[y])
            if len(plot_data) > 0:
                try:
                    sns.boxplot(data=plot_data, x=x, y=y, hue=hue, ax=ax)
                except Exception as e:
                    print(f"  Could not create boxplot for {y}: {e}")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Type', fontsize=8)

        # Plot 1: Predicted vs Population Correlation (by connectome type)
        safe_boxplot(results_df, 'atlas', 'pred_vs_pop_correlation', 'connectome_type', axes[0,0],
                    'Predicted vs Population Correlation', 'Pearson Correlation')
        
        # Plot 2: Predicted vs Population RMSE (by connectome type)
        safe_boxplot(results_df, 'atlas', 'pred_vs_pop_rmse', 'connectome_type', axes[0,1],
                    'Predicted vs Population RMSE', 'RMSE')
        
        # Plot 3: Predicted vs Population LERM (by connectome type)
        safe_boxplot(results_df, 'atlas', 'pred_vs_pop_lerm', 'connectome_type', axes[0,2],
                    'Predicted vs Population LERM', 'LERM')
        
        # Plot 4: Individual vs Population Comparison (if available)
        if 'pred_vs_true_correlation' in results_df.columns and 'true_vs_pop_correlation' in results_df.columns:
            # Create comparison data
            comparison_data = []
            for _, row in results_df.iterrows():
                # Only add if values are not NaN
                if pd.notna(row.get('pred_vs_true_correlation')):
                    comparison_data.append({
                        'atlas': row['atlas'],
                        'connectome_type': row['connectome_type'],
                        'comparison': 'Pred vs True',
                        'correlation': row['pred_vs_true_correlation']
                    })
                if pd.notna(row.get('pred_vs_pop_correlation')):
                    comparison_data.append({
                        'atlas': row['atlas'],
                        'connectome_type': row['connectome_type'],
                        'comparison': 'Pred vs Pop',
                        'correlation': row['pred_vs_pop_correlation']
                    })
                if pd.notna(row.get('true_vs_pop_correlation')):
                    comparison_data.append({
                        'atlas': row['atlas'],
                        'connectome_type': row['connectome_type'],
                        'comparison': 'True vs Pop',
                        'correlation': row['true_vs_pop_correlation']
                    })
            
            comp_df = pd.DataFrame(comparison_data)
            if len(comp_df) > 0:
                try:
                    sns.boxplot(data=comp_df, x='connectome_type', y='correlation', hue='comparison', ax=axes[1,0])
                    axes[1,0].legend(title='Comparison', fontsize=8)
                except Exception as e:
                    print(f"  Could not create comparison boxplot: {e}")
            
            axes[1,0].set_title('Correlation Comparison')
            axes[1,0].set_ylabel('Correlation')
            axes[1,0].tick_params(axis='x', rotation=45)

        
        # Plot 5: Connection count comparison (color by connectome type)
        sns.scatterplot(data=results_df, x='pop_connections', y='pred_connections', 
                       hue='connectome_type', style='atlas', ax=axes[1,1], alpha=0.6)
        axes[1,1].set_title('Connection Count: Population vs Predicted')
        axes[1,1].set_xlabel('Population Average Connections')
        axes[1,1].set_ylabel('Predicted Connections')
        max_val = max(results_df['pop_connections'].max(), results_df['pred_connections'].max())
        axes[1,1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        axes[1,1].legend(title='Type', fontsize=8)
        
        # Plot 6: Distribution of correlation values by connectome type
        for connectome_type in results_df['connectome_type'].unique():
            type_data = results_df[results_df['connectome_type'] == connectome_type]
            if len(type_data) > 0:
                axes[1,2].hist(type_data['pred_vs_pop_correlation'], alpha=0.6, 
                              label=f'{connectome_type}', bins=20)
        axes[1,2].set_title('Distribution of Correlations')
        axes[1,2].set_xlabel('Correlation')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].legend(title='Type', fontsize=8)
        
        plt.tight_layout()
        
        # Save plots
        plot_file = self.output_dir / "population_analysis_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Plots saved to: {plot_file}")
        self.logger.info(f"Comparison plots saved to: {plot_file}")
    
    def _create_population_connectome_plots(self, average_connectomes: Dict[str, Dict[str, np.ndarray]]):
        """
        Create population connectome plots using unified_connectome visualization functions
        
        Args:
            average_connectomes: Nested dictionary {atlas: {connectome_type: average_matrix}}
        """
        print("\nCreating population connectome visualization plots...")
        
        for atlas_name in average_connectomes:
            # Only create plots for NOS (streamline count) to avoid too many plots
            if 'nos' not in average_connectomes[atlas_name]:
                continue
                
            population_connectome = average_connectomes[atlas_name]['nos']
            
            try:
                print(f"  Creating plot for {atlas_name} atlas (NOS)")
                
                # Create a ConnectomeAnalyzer instance for visualization
                analyzer = ConnectomeAnalyzer(
                    predicted_path="",  # Not needed for visualization
                    ground_truth_path="",  # Not needed for visualization
                    atlas_name=atlas_name,
                    output_dir=self.output_dir
                )
                
                # Create a comparison plot with population connectome as both predicted and ground truth
                # This will show the population connectome pattern
                plot_path = analyzer.create_comparison_plot(
                    predicted_connectome=population_connectome,
                    ground_truth_connectome=population_connectome,
                    subject_id=f"population_average_{atlas_name}",
                    title=f"Population Average Connectome - {atlas_name}",
                    save_plot=True
                )
                
                if plot_path:
                    print(f"   Population connectome plot saved: {plot_path}")
                else:
                    print(f"   Could not create plot for {atlas_name}")
                    
            except Exception as e:
                print(f"   Error creating population plot for {atlas_name}: {str(e)}")
                self.logger.error(f"Error creating population plot for {atlas_name}: {str(e)}")
    
    def _create_individual_comparison_plots(self, results_df: pd.DataFrame, 
                                          average_connectomes: Dict[str, Dict[str, np.ndarray]]):
        """
        Create individual subject comparison plots with population average as ground truth
        
        Args:
            results_df: DataFrame with comparison results
            average_connectomes: Nested dictionary {atlas: {connectome_type: average_matrix}}
        """
        print("\nCreating individual subject comparison plots...")
        
        # Select 5 subjects with diverse performance (best, worst, and middle performers)
        sample_subjects = self._select_representative_subjects(results_df, n_subjects=5)
        
        for atlas_name in average_connectomes:
            # Only use NOS for visualization
            if 'nos' not in average_connectomes[atlas_name]:
                continue
                
            population_connectome = average_connectomes[atlas_name]['nos']
            print(f"  Creating comparison plots for {atlas_name} atlas (NOS)")
            
            for i, subject_id in enumerate(sample_subjects):
                try:
                    # Load the predicted connectome for this subject
                    predicted_connectome = self._load_connectome_csv(subject_id, atlas_name, 'nos', "pred")
                    
                    if predicted_connectome is None:
                        # Fallback to labels method
                        predicted_connectome = self._load_subject_connectome(subject_id, atlas_name, "pred")
                    
                    if predicted_connectome is not None:
                        # Create a ConnectomeAnalyzer instance
                        analyzer = ConnectomeAnalyzer(
                            predicted_path="",  # Not needed as we provide matrices directly
                            ground_truth_path="",  # Not needed as we provide matrices directly
                            atlas_name=atlas_name,
                            output_dir=self.output_dir
                        )
                        
                        # Create comparison plot with population average as ground truth
                        plot_path = analyzer.create_comparison_plot(
                            predicted_connectome=predicted_connectome,
                            ground_truth_connectome=population_connectome,
                            subject_id=f"{subject_id}_vs_population_{atlas_name}",
                            title=f"Subject {subject_id} vs Population Average - {atlas_name}",
                            save_plot=True
                        )
                        
                        if plot_path:
                            print(f"   Comparison plot created for subject {subject_id}: {plot_path}")
                        else:
                            print(f"   Could not create comparison plot for subject {subject_id}")
                    else:
                        print(f"   Could not load predicted connectome for subject {subject_id}")
                        
                except Exception as e:
                    print(f"   Error creating comparison plot for subject {subject_id}: {str(e)}")
                    self.logger.error(f"Error creating comparison plot for subject {subject_id}: {str(e)}")
    
    def _select_representative_subjects(self, results_df: pd.DataFrame, n_subjects: int = 5) -> List[str]:
        """
        Select representative subjects with diverse performance levels
        
        Args:
            results_df: DataFrame with comparison results
            n_subjects: Number of subjects to select
            
        Returns:
            List of subject IDs
        """
        if len(results_df) < n_subjects:
            return results_df['subject_id'].tolist()
        
        # Calculate average correlation across atlases for each subject
        correlation_cols = [col for col in results_df.columns if 'correlation' in col]
        if correlation_cols:
            results_df['avg_correlation'] = results_df[correlation_cols].mean(axis=1)
            
            # Sort by average correlation
            sorted_df = results_df.sort_values('avg_correlation')
            
            # Select subjects with diverse performance
            n_total = len(sorted_df)
            indices = [
                0,  # Worst performer
                n_total // 4,  # Lower quartile
                n_total // 2,  # Median
                3 * n_total // 4,  # Upper quartile
                n_total - 1  # Best performer
            ]
            
            selected_subjects = []
            for idx in indices[:n_subjects]:
                if idx < len(sorted_df):
                    selected_subjects.append(sorted_df.iloc[idx]['subject_id'])
            
            return selected_subjects
        else:
            # Fallback: select first n subjects
            return results_df['subject_id'].head(n_subjects).tolist()
    
    def _save_metrics_json(self, results_df: pd.DataFrame, average_connectomes: Dict[str, np.ndarray]):
        """
        Save comprehensive metrics as JSON file similar to efficient_multi_subject_analysis.py
        
        Args:
            results_df: DataFrame with comparison results
            average_connectomes: Dictionary containing population average connectomes
        """
        print("\nSaving metrics as JSON...")
        
        try:
            # Prepare metrics dictionary
            metrics = {
                'analysis_info': {
                    'analysis_type': 'population_connectome_analysis',
                    'timestamp': datetime.now().isoformat(),
                    'n_training_subjects': len(self.train_subjects),
                    'n_test_subjects': len(results_df),
                    'atlases': list(average_connectomes.keys())
                },
                'population_statistics': {},
                'subject_results': [],
                'summary_statistics': {}
            }
            
            # Add population statistics
            for atlas_name, connectome_dict in average_connectomes.items():
                metrics['population_statistics'][atlas_name] = {}
                for connectome_type, connectome in connectome_dict.items():
                    metrics['population_statistics'][atlas_name][connectome_type] = {
                        'shape': connectome.shape,
                        'mean_connectivity': float(np.mean(connectome)),
                        'std_connectivity': float(np.std(connectome)),
                        'min_connectivity': float(np.min(connectome)),
                        'max_connectivity': float(np.max(connectome)),
                        'total_connections': int(np.sum(connectome > 0))
                    }
            
            # Add individual subject results
            for _, row in results_df.iterrows():
                subject_result = {
                    'subject_id': row['subject_id']
                }
                
                # Add all metrics for this subject
                for col in results_df.columns:
                    if col != 'subject_id' and not isinstance(row[col], (pd.Series, pd.DataFrame)):
                         # Handle numpy types by converting to native python types
                         val = row[col]
                         if hasattr(val, 'item'):
                             val = val.item()
                         subject_result[col] = val if pd.notna(val) else None
                
                metrics['subject_results'].append(subject_result)
            
            # Add summary statistics
            for col in results_df.columns:
                if col != 'subject_id' and results_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    metrics['summary_statistics'][col] = {
                        'mean': float(results_df[col].mean()),
                        'std': float(results_df[col].std()),
                        'median': float(results_df[col].median()),
                        'min': float(results_df[col].min()),
                        'max': float(results_df[col].max()),
                        'q25': float(results_df[col].quantile(0.25)),
                        'q75': float(results_df[col].quantile(0.75))
                    }
            
            # Save JSON file
            json_path = os.path.join(self.output_dir, 'population_connectome_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"   Metrics saved to: {json_path}")
            self.logger.info(f"Metrics saved to JSON: {json_path}")
            
        except Exception as e:
            print(f"   Error saving metrics JSON: {str(e)}")
            self.logger.error(f"Error saving metrics JSON: {str(e)}")
            traceback.print_exc()

    def _create_connectome_visualizations(self, average_connectomes: Dict[str, Dict[str, np.ndarray]]):
        """
        Create simple heatmap visualizations of population average connectomes
        
        Args:
            average_connectomes: Nested dictionary {atlas: {connectome_type: average_matrix}}
        """
        print("\nCreating connectome heatmap visualizations...")
        
        for atlas_name in average_connectomes:
            for connectome_type in average_connectomes[atlas_name]:
                connectome = average_connectomes[atlas_name][connectome_type]
                
                try:
                    print(f"   Creating heatmap for {atlas_name} {connectome_type.upper()}")
                    
                    # Create figure
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                    fig.suptitle(f'Population Average Connectome - {atlas_name} ({connectome_type.upper()})', 
                               fontsize=14, fontweight='bold')
                    
                    # Plot 1: Full connectome heatmap
                    im1 = axes[0].imshow(connectome, cmap='viridis', aspect='auto')
                    axes[0].set_title('Full Connectome')
                    axes[0].set_xlabel('Brain Region')
                    axes[0].set_ylabel('Brain Region')
                    cbar_label = 'Average Number of Streamlines' if connectome_type == 'nos' else f'Average {connectome_type.upper()}'
                    plt.colorbar(im1, ax=axes[0], label=cbar_label)
                    
                    # Plot 2: Log-scale connectome (non-zero values only)
                    connectome_log = np.log(connectome + 1e-10)  # Add epsilon to handle zeros
                    connectome_log = np.nan_to_num(connectome_log, neginf=0) # Handle log(0)
                
                    im2 = axes[1].imshow(connectome_log, cmap='viridis', aspect='auto')
                    axes[1].set_title('Log-scale Connectome')
                    axes[1].set_xlabel('Brain Region')
                    axes[1].set_ylabel('Brain Region')
                    plt.colorbar(im2, ax=axes[1], label=f'Log({cbar_label})')
                    
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = self.output_dir / f'population_connectome_{atlas_name}_{connectome_type}_heatmap.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   Heatmap saved: {plot_path.name}")
                    
                except Exception as e:
                    print(f"   Error creating heatmap for {atlas_name} {connectome_type}: {str(e)}")
                    self.logger.error(f"Error creating heatmap for {atlas_name} {connectome_type}: {str(e)}")
    
    def _create_subject_wise_report(self, results_df: pd.DataFrame):
        """Create detailed subject-wise report"""
        print("\nCreating subject-wise report...")
        
        if len(results_df) == 0:
            print("  No results to report")
            return
        
        # Sort by correlation for each atlas and connectome type combination
        for atlas in self.atlases:
            for connectome_type in self.connectome_types:
                # Filter data for this combination
                mask = (results_df['atlas'] == atlas) & (results_df['connectome_type'] == connectome_type)
                data = results_df[mask].copy()
                
                if len(data) == 0:
                    continue
                
                # Sort by correlation (best to worst)
                data_sorted = data.sort_values('pred_vs_pop_correlation', ascending=False)
                
                # Save sorted results
                atlas_file = self.output_dir / f"subject_results_{atlas}_{connectome_type}.csv"
                data_sorted.to_csv(atlas_file, index=False)
                
                # Create top/bottom performers summary
                n_show = min(5, len(data_sorted))
                
                print(f"\n{atlas} ({connectome_type.upper()}) - Top {n_show} performers:")
                top_performers = data_sorted.head(n_show)[['subject_id', 'pred_vs_pop_correlation', 'pred_vs_pop_lerm']]
                print(top_performers.to_string(index=False))
        
        self.logger.info("Subject-wise report created")
    
    def run_full_analysis(self, force_recompute: bool = False):
        """
        Run the complete population connectome analysis pipeline
        
        Args:
            force_recompute: Whether to recompute cached results
        """
        print("="*80)
        print("POPULATION CONNECTOME ANALYSIS")
        print("="*80)
        print(f"Training subjects: {len(self.train_subjects)}")
        print(f"Test subjects: {len(self.test_subjects)}")
        print(f"Atlases: {self.atlases}")
        print("="*80)
        
        self.logger.info("="*80)
        self.logger.info("POPULATION CONNECTOME ANALYSIS")
        self.logger.info(f"Training subjects: {len(self.train_subjects)}")
        self.logger.info(f"Test subjects: {len(self.test_subjects)}")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Compute population average connectomes
            average_connectomes = self.compute_population_average_connectomes(force_recompute=force_recompute)
            
            if not average_connectomes:
                print("Failed to compute population average connectomes!")
                return
            
            # Step 2: Compare test subjects to population averages
            results_df = self.compare_test_subjects_to_population(average_connectomes)
            
            if len(results_df) == 0:
                print("No successful comparisons!")
                return
            
            # Step 3: Create comprehensive report
            self.create_population_analysis_report(results_df, average_connectomes)
            
            # Final summary
            elapsed_time = time.time() - start_time
            print(f"\nPopulation analysis completed successfully!")
            print(f"Analyzed {len(results_df)} subject-atlas-type combinations")
            print(f"Total analysis time: {elapsed_time:.2f} seconds")
            print(f"Results saved to: {self.output_dir}")
            
            self.logger.info("Population analysis completed successfully!")
            self.logger.info(f"Analyzed {len(results_df)} subject-atlas-type combinations")
            self.logger.info(f"Total analysis time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"\nAnalysis failed with error: {e}")
            self.logger.error(f"Analysis failed: {e}")
            traceback.print_exc()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Population Connectome Analysis')
    parser.add_argument('--base-path', default='/media/volume/MV_HCP', 
                       help='Base path for data (default: /media/volume/MV_HCP)')
    parser.add_argument('--compute-average', action='store_true',
                       help='Compute population average connectomes only')
    parser.add_argument('--test-comparison', action='store_true',
                       help='Compare test subjects to population averages only')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Run complete analysis pipeline')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation of cached results')
    
    args = parser.parse_args()
    
    # Default to full analysis if no specific mode selected
    if not any([args.compute_average, args.test_comparison, args.full_analysis]):
        args.full_analysis = True
    
    # Initialize analysis
    analysis = PopulationConnectomeAnalysis(base_path=args.base_path)
    
    try:
        if args.full_analysis:
            analysis.run_full_analysis(force_recompute=args.force_recompute)
        elif args.compute_average:
            average_connectomes = analysis.compute_population_average_connectomes(
                force_recompute=args.force_recompute)
            n_atlases = len(average_connectomes)
            n_types = sum(len(types_dict) for types_dict in average_connectomes.values())
            print(f"Population average connectomes computed: {n_atlases} atlases x {n_types} types")
        elif args.test_comparison:
            # Load existing average connectomes
            average_connectomes = {}
            for atlas in analysis.atlases:
                average_connectomes[atlas] = {}
                for connectome_type in analysis.connectome_types:
                    cache_file = analysis.output_dir / f"population_average_{atlas}_{connectome_type}.npy"
                    if cache_file.exists():
                        average_connectomes[atlas][connectome_type] = np.load(cache_file)
                        print(f"Loaded cached average connectome for {atlas} {connectome_type}")
                    else:
                        print(f"No cached average connectome found for {atlas} {connectome_type}")
            
            # Check if any connectomes were loaded
            total_loaded = sum(len(types_dict) for types_dict in average_connectomes.values())
            
            if total_loaded > 0:
                results_df = analysis.compare_test_subjects_to_population(average_connectomes)
                print(f"Compared {len(results_df)} subject-atlas-type combinations")
            else:
                print("No average connectomes available. Run --compute-average first.")
                
    except Exception as e:
        print(f"Analysis failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()