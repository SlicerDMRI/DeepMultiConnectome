"""
Unified Connectome Analysis Module

This module provides a clean, standardized way to handle all connectome operations:
- Creating connectomes from labels and diffusion metrics
- Computing comprehensive metrics and comparisons
- Generating visualizations and reports
- Network analysis

Replaces the old metrics_connectome.py with a cleaner, more extensible design.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Statistical and ML imports
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    mean_squared_error, mean_absolute_error, classification_report
)
from numpy.linalg import norm

# Network analysis
import networkx as nx

# Local imports
from tractography.label_encoder import convert_labels_list
from utils.connectome_config import ATLAS_CONFIG, DIFFUSION_METRICS, CONNECTOME_TYPES


class ConnectomeAnalyzer:
    """
    Unified class for comprehensive connectome analysis
    
    This class handles:
    - Connectome creation from labels and weights
    - Statistical comparisons between connectomes
    - Network analysis
    - Visualization and reporting
    """
    
    def __init__(self, 
                 atlas: str = "aparc+aseg",
                 out_path: str = "output",
                 logger = None):
        """
        Initialize the ConnectomeAnalyzer
        
        Args:
            atlas: Atlas name (aparc+aseg or aparc.a2009s+aseg)
            out_path: Output directory for results
            logger: Logger instance
        """
        self.atlas = atlas
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Atlas configuration
        if atlas not in ATLAS_CONFIG:
            raise ValueError(f"Unknown atlas: {atlas}. Available: {list(ATLAS_CONFIG.keys())}")
        
        self.num_labels = ATLAS_CONFIG[atlas]['num_labels']
        self.atlas_description = ATLAS_CONFIG[atlas]['description']
        
        # Storage for connectomes and results
        self.connectomes = {}
        self.metrics = {}
        self.network_metrics = {}
        
    def _log(self, message: str, level: str = "info"):
        """Log message if logger is available"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def _load_metric_values_improved(self, metric_file: Path, expected_length: int, metric_name: str) -> np.ndarray:
        """
        Improved metric loading that maintains consistent array sizes by preserving NaN positions
        
        Args:
            metric_file: Path to the metric file
            expected_length: Expected number of values
            metric_name: Name of the metric for logging
            
        Returns:
            NumPy array of metric values with NaN for missing/invalid values
        """
        try:
            with open(metric_file, 'r') as f:
                lines = f.readlines()
            
            # Check if it's a single-line file (space-separated) or multi-line file (newline-separated)
            all_values = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Try to split by spaces first (original format)
                    line_values = line.split()
                    all_values.extend(line_values)
            
            if not all_values:
                self._log(f"No data found in {metric_name} file", "warning")
                return np.full(expected_length, np.nan)
            
            values = all_values
            
            # Create output array filled with NaN
            result = np.full(expected_length, np.nan)
            
            valid_count = 0
            nan_count = 0
            invalid_count = 0
            
            # Parse values maintaining positions
            max_values = min(len(values), expected_length)
            
            # CRITICAL FIX: If the metric file is significantly shorter than expected (e.g. corruption),
            # discard it entirely rather than populating only the first few streamlines.
            if len(values) < expected_length * 0.5 and len(values) > 0:
                self._log(f"Metric {metric_name}: CRITICAL - File has {len(values)} values, expected {expected_length}. Discarding to avoid bias.", "warning")
                return np.full(expected_length, np.nan)

            for i in range(max_values):
                val = values[i]
                if val.lower() == 'nan':
                    result[i] = np.nan
                    nan_count += 1
                else:
                    try:
                        result[i] = float(val)
                        valid_count += 1
                    except ValueError:
                        result[i] = np.nan
                        invalid_count += 1
            
            # Log statistics
            total_in_file = len(values)
            if total_in_file != expected_length:
                self._log(f"Metric {metric_name}: file has {total_in_file} values, expected {expected_length}", "info")
            
            if nan_count > 0 or invalid_count > 0:
                self._log(f"Metric {metric_name}: {valid_count} valid, {nan_count} NaN, {invalid_count} invalid values", "info")
            else:
                self._log(f"Metric {metric_name}: {valid_count} valid values loaded", "info")
            
            return result
            
        except Exception as e:
            self._log(f"Error loading {metric_name} values: {e}", "error")
            return np.full(expected_length, np.nan)
    
    def create_connectome_from_labels(self, 
                                    labels: Union[List, np.ndarray],
                                    weights: Optional[Union[List, np.ndarray]] = None,
                                    encoding: str = 'symmetric',
                                    connectome_name: str = None,
                                    aggregation_method: str = 'mean') -> np.ndarray:
        """
        Create connectome matrix from streamline labels
        
        Args:
            labels: List/array of streamline labels
            weights: Optional weights for each streamline (e.g., FA values, SIFT2 weights)
            encoding: Label encoding type ('symmetric', 'asymmetric')
            connectome_name: Name to store the connectome
            aggregation_method: How to aggregate weights ('mean' or 'sum')
                               - 'mean': Average weights for each connection (default for diffusion metrics)
                               - 'sum': Sum weights for each connection (recommended for SIFT2 weights)
            
        Returns:
            Connectome matrix
        """
        if isinstance(labels, list):
            labels = np.array(labels)
            
        # Decode labels if they're encoded
        if encoding == 'symmetric':
            try:
                decoded_labels = convert_labels_list(
                    labels, encoding_type=encoding, 
                    mode='decode', num_labels=self.num_labels
                )
            except:
                # Assume labels are already decoded
                decoded_labels = labels
        else:
            decoded_labels = labels
        
        # Initialize connectome matrix
        connectome_matrix = np.zeros((self.num_labels, self.num_labels))
        
        # Build connectome
        if weights is None:
            # Count-based connectome
            for label_pair in decoded_labels:
                if isinstance(label_pair, (list, tuple)) and len(label_pair) == 2:
                    i, j = int(label_pair[0]), int(label_pair[1])
                    if 0 <= i < self.num_labels and 0 <= j < self.num_labels:
                        connectome_matrix[i, j] += 1
                        if i != j:  # Symmetric filling
                            connectome_matrix[j, i] += 1
        else:
            # Weighted connectome
            if len(weights) != len(decoded_labels):
                raise ValueError("Weights and labels must have the same length")
                
            # Track counts for averaging (only needed for mean aggregation)
            count_matrix = np.zeros((self.num_labels, self.num_labels))
            
            for idx, label_pair in enumerate(decoded_labels):
                if isinstance(label_pair, (list, tuple)) and len(label_pair) == 2:
                    i, j = int(label_pair[0]), int(label_pair[1])
                    if 0 <= i < self.num_labels and 0 <= j < self.num_labels and idx < len(weights):
                        weight = float(weights[idx])
                        if not np.isnan(weight) and not np.isinf(weight):
                            connectome_matrix[i, j] += weight
                            count_matrix[i, j] += 1
                            if i != j:
                                connectome_matrix[j, i] += weight
                                count_matrix[j, i] += 1
            
            # Apply aggregation method
            if aggregation_method.lower() == 'mean':
                # Average the weights
                with np.errstate(divide='ignore', invalid='ignore'):
                    connectome_matrix = np.divide(connectome_matrix, count_matrix, 
                                                out=np.zeros_like(connectome_matrix), 
                                                where=count_matrix!=0)
            elif aggregation_method.lower() == 'sum':
                # Keep summed weights (no additional processing needed)
                pass
            else:
                self._log(f"Warning: Unknown aggregation method '{aggregation_method}'. Using 'mean'.", "warning")
                # Default to mean
                with np.errstate(divide='ignore', invalid='ignore'):
                    connectome_matrix = np.divide(connectome_matrix, count_matrix, 
                                                out=np.zeros_like(connectome_matrix), 
                                                where=count_matrix!=0)
        
        # Remove the first row and column (background label)
        connectome_matrix = connectome_matrix[1:, 1:]
        
        # Store connectome if name provided
        if connectome_name:
            self.connectomes[connectome_name] = connectome_matrix
            agg_desc = f" (weights {aggregation_method})" if weights is not None else ""
            self._log(f"Created connectome '{connectome_name}' with shape {connectome_matrix.shape}{agg_desc}")
        
        return connectome_matrix
    
    def load_connectome_from_file(self,
                                filepath: Union[str, Path], 
                                connectome_name: str) -> np.ndarray:
        """
        Load connectome from CSV file
        
        Args:
            filepath: Path to connectome CSV file
            connectome_name: Name to store the connectome
            
        Returns:
            Connectome matrix
        """
        try:
            connectome_matrix = np.loadtxt(filepath, delimiter=',')
            self.connectomes[connectome_name] = connectome_matrix
            self._log(f"Loaded connectome '{connectome_name}' from {filepath}")
            return connectome_matrix
        except Exception as e:
            self._log(f"Error loading connectome from {filepath}: {e}", "error")
            return None
    
    def save_connectome(self, 
                       connectome_name: str, 
                       filename: Optional[str] = None) -> str:
        """
        Save connectome to CSV file
        
        Args:
            connectome_name: Name of stored connectome
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if connectome_name not in self.connectomes:
            raise ValueError(f"Connectome '{connectome_name}' not found")
        
        if filename is None:
            filename = f"connectome_{connectome_name}_{self.atlas}.csv"
        
        output_path = self.out_path / filename
        np.savetxt(output_path, self.connectomes[connectome_name], 
                  delimiter=',', fmt='%.6f')
        
        self._log(f"Saved connectome '{connectome_name}' to {output_path}")
        return str(output_path)
    
    def compute_comparison_metrics(self, 
                                 connectome1_name: str, 
                                 connectome2_name: str,
                                 comparison_name: Optional[str] = None) -> Dict:
        """
        Compute comprehensive comparison metrics between two connectomes
        
        Args:
            connectome1_name: Name of first connectome (typically ground truth)
            connectome2_name: Name of second connectome (typically predicted)
            comparison_name: Name for this comparison
            
        Returns:
            Dictionary of comparison metrics
        """
        if connectome1_name not in self.connectomes:
            raise ValueError(f"Connectome '{connectome1_name}' not found")
        if connectome2_name not in self.connectomes:
            raise ValueError(f"Connectome '{connectome2_name}' not found")
        
        true_conn = self.connectomes[connectome1_name]
        pred_conn = self.connectomes[connectome2_name]
        
        if comparison_name is None:
            comparison_name = f"{connectome1_name}_vs_{connectome2_name}"
        
        # Flatten matrices for correlation analysis (include diagonal, consistent with other scripts)
        # Use upper triangular including diagonal, same as other evaluation scripts
        triu_mask = np.triu_indices_from(true_conn, k=0)
        true_flat = true_conn[triu_mask]
        pred_flat = pred_conn[triu_mask]
        
        # Remove zero connections for some metrics
        nonzero_mask = (true_flat != 0) | (pred_flat != 0)
        true_nonzero = true_flat[nonzero_mask]
        pred_nonzero = pred_flat[nonzero_mask]
        
        metrics = {}
        
        # Correlation metrics (using same methods as other evaluation scripts)
        try:
            # Use np.corrcoef for consistency with other scripts
            correlation_matrix = np.corrcoef(true_flat, pred_flat)
            pearson_r = correlation_matrix[0, 1]
            metrics['pearson_r'] = pearson_r
            
            # Also compute using scipy for p-value
            pearson_r_scipy, pearson_p = pearsonr(true_flat, pred_flat)
            metrics['pearson_p'] = pearson_p
        except:
            metrics['pearson_r'] = np.nan
            metrics['pearson_p'] = np.nan
        
        try:
            spearman_r, spearman_p = spearmanr(true_flat, pred_flat)
            metrics['spearman_r'] = spearman_r
            metrics['spearman_p'] = spearman_p
        except:
            metrics['spearman_r'] = np.nan
            metrics['spearman_p'] = np.nan
        
        # Error metrics
        metrics['mse'] = mean_squared_error(true_flat, pred_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(true_flat, pred_flat)
        
        # Relative error metrics
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = np.abs(true_flat - pred_flat) / (np.abs(true_flat) + np.abs(pred_flat) + 1e-10)
            metrics['mean_relative_error'] = np.mean(relative_error)
        
        # LERM (Log-Euclidean Relative Magnitude) - using improved numerical stability
        try:
            from scipy.linalg import logm
            import warnings
            
            # Check if matrices are suitable for matrix logarithm
            true_min = np.min(true_conn)
            pred_min = np.min(pred_conn)
            true_max = np.max(true_conn)
            pred_max = np.max(pred_conn)
            
            # Use matrix logarithm only if matrices are well-conditioned
            if (true_min >= 0 and pred_min >= 0 and 
                true_max > 1e-6 and pred_max > 1e-6 and
                np.linalg.cond(true_conn) < 1e12 and np.linalg.cond(pred_conn) < 1e12):
                
                # Add appropriate epsilon based on matrix scale
                epsilon = max(1e-10, min(true_max, pred_max) * 1e-6)
                true_conn_safe = true_conn + epsilon
                pred_conn_safe = pred_conn + epsilon
                
                # Suppress specific warnings during logm computation
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    warnings.filterwarnings('ignore', message='.*nearly singular.*')
                    warnings.filterwarnings('ignore', message='.*divide by zero.*')
                    warnings.filterwarnings('ignore', message='.*invalid value.*')
                    
                    try:
                        logm_true = logm(true_conn_safe)
                        logm_pred = logm(pred_conn_safe)
                        
                        # Check if logm results are valid (no NaN/inf)
                        if (np.isfinite(logm_true).all() and np.isfinite(logm_pred).all()):
                            metrics['mean_lerm'] = norm(logm_pred - logm_true, 'fro')
                        else:
                            raise ValueError("Matrix logarithm produced non-finite values")
                            
                    except (np.linalg.LinAlgError, ValueError):
                        raise ValueError("Matrix logarithm computation failed")
            else:
                raise ValueError("Matrices not suitable for matrix logarithm")
                
        except (ImportError, ValueError, Exception):
            # Use fallback LERM computation (standard relative difference)
            with np.errstate(divide='ignore', invalid='ignore'):
                denominator = true_conn + pred_conn
                lerm_matrix = np.zeros_like(true_conn)
                valid_mask = denominator != 0
                lerm_matrix[valid_mask] = (2 * np.abs(true_conn[valid_mask] - pred_conn[valid_mask]) / 
                                         denominator[valid_mask])
                # Handle any remaining NaN/inf values
                lerm_matrix = np.where(np.isfinite(lerm_matrix), lerm_matrix, 0)
                metrics['mean_lerm'] = np.mean(lerm_matrix[triu_mask])
        
        # Matrix norms
        diff_matrix = true_conn - pred_conn
        metrics['frobenius_norm'] = norm(diff_matrix, 'fro')
        metrics['max_absolute_error'] = np.max(np.abs(diff_matrix))
        
        # Earth Mover's Distance (Wasserstein)
        try:
            metrics['wasserstein_distance'] = wasserstein_distance(true_flat, pred_flat)
        except:
            metrics['wasserstein_distance'] = np.nan
        
        # Edge overlap metrics
        true_binary = (true_conn > 0).astype(int)
        pred_binary = (pred_conn > 0).astype(int)
        
        true_edges = np.sum(true_binary[triu_mask])
        pred_edges = np.sum(pred_binary[triu_mask])
        overlap_edges = np.sum((true_binary & pred_binary)[triu_mask])
        
        metrics['edge_overlap'] = overlap_edges / max(true_edges, pred_edges, 1)
        metrics['jaccard_index'] = overlap_edges / max(true_edges + pred_edges - overlap_edges, 1)
        
        # Connection strength statistics
        metrics['true_total_strength'] = np.sum(true_conn)
        metrics['pred_total_strength'] = np.sum(pred_conn)
        metrics['strength_ratio'] = (metrics['pred_total_strength'] / 
                                   max(metrics['true_total_strength'], 1e-10))
        
        # Store metrics
        self.metrics[comparison_name] = metrics
        
        self._log(f"Computed comparison metrics for '{comparison_name}'")
        self._log(f"  Pearson r: {metrics['pearson_r']:.4f}")
        self._log(f"  Spearman r: {metrics['spearman_r']:.4f}")
        self._log(f"  RMSE: {metrics['rmse']:.4f}")
        self._log(f"  Mean LERM: {metrics['mean_lerm']:.4f}")
        
        return metrics
    
    def compute_network_metrics(self, connectome_name: str, 
                              compute_advanced: bool = False,
                              compute_centrality: bool = False,
                              compute_community: bool = False) -> Dict:
        """
        Compute network analysis metrics with optional advanced computations
        
        Args:
            connectome_name: Name of connectome to analyze
            compute_advanced: Whether to compute advanced metrics (path length, efficiency)
            compute_centrality: Whether to compute centrality measures (slow for large networks)
            compute_community: Whether to compute community detection metrics
            
        Returns:
            Dictionary of network metrics
        """
        if connectome_name not in self.connectomes:
            raise ValueError(f"Connectome '{connectome_name}' not found")
        
        matrix = self.connectomes[connectome_name]
        
        # Create networkx graph
        G = nx.from_numpy_array(matrix)
        
        # Remove isolated nodes for some calculations
        largest_cc = max(nx.connected_components(G), key=len) if nx.number_connected_components(G) > 0 else set()
        G_connected = G.subgraph(largest_cc).copy() if len(largest_cc) > 1 else G
        
        metrics = {}
        
        try:
            # Basic network properties (always computed - fast)
            metrics['num_nodes'] = G.number_of_nodes()
            metrics['num_edges'] = G.number_of_edges()
            metrics['density'] = nx.density(G)
            metrics['num_connected_components'] = nx.number_connected_components(G)
            
            # Basic clustering (fast)
            metrics['average_clustering'] = nx.average_clustering(G, weight='weight')
            
            # Advanced metrics (slower - only if requested)
            if compute_advanced:
                # Global metrics (require connected graph)
                if nx.is_connected(G_connected) and len(G_connected) > 1:
                    # Transform weights for path-based metrics (shorter paths = stronger connections)
                    with np.errstate(divide='ignore'):
                        weight_matrix = 1.0 / matrix
                    weight_matrix[np.isinf(weight_matrix)] = 0
                    G_weighted = nx.from_numpy_array(weight_matrix)
                    G_weighted_connected = G_weighted.subgraph(largest_cc).copy()
                    
                    if nx.is_connected(G_weighted_connected):
                        metrics['average_path_length'] = nx.average_shortest_path_length(G_weighted_connected, weight='weight')
                        metrics['global_efficiency'] = nx.global_efficiency(G_weighted_connected)
                    else:
                        metrics['average_path_length'] = np.nan
                        metrics['global_efficiency'] = np.nan
                else:
                    metrics['average_path_length'] = np.nan
                    metrics['global_efficiency'] = np.nan
                
                # Local efficiency (can be slow)
                metrics['local_efficiency'] = nx.local_efficiency(G)
            else:
                # Set to NaN if not computed
                metrics['average_path_length'] = np.nan
                metrics['global_efficiency'] = np.nan
                metrics['local_efficiency'] = np.nan
            
            # Centrality measures (very slow for large networks)
            if compute_centrality:
                degree_cent = nx.degree_centrality(G)
                metrics['average_degree_centrality'] = np.mean(list(degree_cent.values()))
                
                if len(G_connected) > 2:
                    try:
                        betweenness_cent = nx.betweenness_centrality(G_connected, weight='weight')
                        metrics['average_betweenness_centrality'] = np.mean(list(betweenness_cent.values()))
                    except:
                        metrics['average_betweenness_centrality'] = np.nan
                    
                    try:
                        eigenvector_cent = nx.eigenvector_centrality(G_connected, weight='weight', max_iter=1000)
                        metrics['average_eigenvector_centrality'] = np.mean(list(eigenvector_cent.values()))
                    except:
                        metrics['average_eigenvector_centrality'] = np.nan
                else:
                    metrics['average_betweenness_centrality'] = np.nan
                    metrics['average_eigenvector_centrality'] = np.nan
            else:
                # Set to NaN if not computed
                metrics['average_degree_centrality'] = np.nan
                metrics['average_betweenness_centrality'] = np.nan
                metrics['average_eigenvector_centrality'] = np.nan
            
            # Assortativity (medium speed)
            try:
                metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight')
            except:
                metrics['degree_assortativity'] = np.nan
            
            # Community detection (can be slow)
            if compute_community:
                try:
                    communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
                    metrics['modularity'] = nx.algorithms.community.modularity(G, communities, weight='weight')
                    metrics['num_communities'] = len(communities)
                except:
                    metrics['modularity'] = np.nan
                    metrics['num_communities'] = np.nan
            else:
                # Set to NaN if not computed
                metrics['modularity'] = np.nan
                metrics['num_communities'] = np.nan
            
            # Small-world properties (requires path length - only if advanced computed)
            if compute_advanced:
                try:
                    if (metrics['average_clustering'] > 0 and not np.isnan(metrics['average_path_length']) 
                        and metrics['average_path_length'] > 0):
                        # Create random reference graph
                        random_G = nx.random_reference(G, niter=3)
                        if nx.is_connected(random_G):
                            L_rand = nx.average_shortest_path_length(random_G)
                            C_rand = nx.average_clustering(random_G)
                            
                            if C_rand > 0 and L_rand > 0:
                                metrics['small_world_sigma'] = (metrics['average_clustering'] / C_rand) / (metrics['average_path_length'] / L_rand)
                            else:
                                metrics['small_world_sigma'] = np.nan
                        else:
                            metrics['small_world_sigma'] = np.nan
                    else:
                        metrics['small_world_sigma'] = np.nan
                except:
                    metrics['small_world_sigma'] = np.nan
            else:
                metrics['small_world_sigma'] = np.nan
            
        except Exception as e:
            self._log(f"Error computing network metrics: {e}", "warning")
            # Fill with NaNs if computation fails
            metric_names = [
                'num_nodes', 'num_edges', 'density', 'num_connected_components',
                'average_path_length', 'global_efficiency', 'average_clustering',
                'local_efficiency', 'average_degree_centrality', 'average_betweenness_centrality',
                'average_eigenvector_centrality', 'degree_assortativity', 'modularity',
                'num_communities', 'small_world_sigma'
            ]
            for name in metric_names:
                if name not in metrics:
                    metrics[name] = np.nan
        
        # Store network metrics
        self.network_metrics[connectome_name] = metrics
        
        # Log what was computed
        computed_types = []
        if compute_advanced:
            computed_types.append("advanced")
        if compute_centrality:
            computed_types.append("centrality")
        if compute_community:
            computed_types.append("community")
        
        computed_str = f" ({', '.join(computed_types)})" if computed_types else " (basic only)"
        self._log(f"Computed network metrics for '{connectome_name}'{computed_str}")
        return metrics
    
    def _plot_single_connectome(self, 
                              connectome_matrix: np.ndarray, 
                              title: str, 
                              log_scale: bool = True, 
                              difference: Union[bool, str] = False) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single connectome matrix with proper scaling and formatting
        
        Args:
            connectome_matrix: The connectome matrix to plot
            title: Title for the plot
            log_scale: Whether to use logarithmic scaling
            difference: Type of difference plot ('percent', 'accuracy', True for diff, False for normal)
            
        Returns:
            Tuple of (figure, axes) objects
        """
        # Determine colormap and normalization based on plot type
        if difference == True:
            if not log_scale:
                cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
                norm = mcolors.TwoSlopeNorm(
                    vmin=connectome_matrix.min(), 
                    vcenter=0, 
                    vmax=connectome_matrix.max()
                )
            else:
                cmap = plt.get_cmap('Reds')
                norm = mcolors.LogNorm(
                    vmin=max(connectome_matrix.min(), 1), 
                    vmax=connectome_matrix.max()
                )
                connectome_matrix = np.where(connectome_matrix == 0, 1e-6, connectome_matrix)
        elif difference == 'percent':
            cmap = plt.get_cmap('RdBu_r')
            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        elif difference == 'accuracy':
            cmap = plt.get_cmap('BuGn')
            norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        else:
            # Normal connectome plot
            if log_scale and np.any(connectome_matrix > 0):
                cmap = plt.get_cmap('viridis')
                norm = mcolors.LogNorm(
                    vmin=max(connectome_matrix[connectome_matrix > 0].min(), 1e-6), 
                    vmax=connectome_matrix.max()
                )
                # Handle zero values by replacing them with a small positive value for log scale
                connectome_matrix = np.where(connectome_matrix == 0, 1e-6, connectome_matrix)
            else:
                cmap = plt.get_cmap('viridis')
                norm = None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the connectome matrix with square aspect ratio
        im = ax.imshow(connectome_matrix, cmap=cmap, norm=norm, aspect='equal', origin='upper')
        
        # Add colorbar
        if difference == True:
            cbar_label = 'Connection Difference'
        elif difference == 'percent':
            cbar_label = 'Percent Change'
        elif difference == 'accuracy':
            cbar_label = 'Accuracy'
        else:
            cbar_label = 'Connection Strength'
            
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Region Index', fontsize=12)
        ax.set_ylabel('Region Index', fontsize=12)
        
        # Set tick marks - show every 10th region for clarity
        num_nodes = connectome_matrix.shape[0]
        if num_nodes > 20:  # Only add ticks for larger matrices
            tick_positions = np.arange(9, num_nodes, 10)
            tick_labels = np.arange(10, num_nodes + 1, 10)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
        
        # Ensure square appearance
        ax.set_aspect('equal', adjustable='box')
        
        return fig, ax
    
    def create_comparison_plot(self,
                             connectome1_name: str, 
                             connectome2_name: str,
                             plot_name: Optional[str] = None,
                             figsize: Tuple[int, int] = (20, 12),
                             log_scale: bool = True) -> str:
        """
        Create comprehensive comparison plot between two connectomes
        
        Args:
            connectome1_name: Name of first connectome
            connectome2_name: Name of second connectome  
            plot_name: Name for the plot file
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        if connectome1_name not in self.connectomes:
            raise ValueError(f"Connectome '{connectome1_name}' not found")
        if connectome2_name not in self.connectomes:
            raise ValueError(f"Connectome '{connectome2_name}' not found")
        
        true_conn = self.connectomes[connectome1_name]
        pred_conn = self.connectomes[connectome2_name]
        
        if plot_name is None:
            plot_name = f"comparison_{connectome1_name}_vs_{connectome2_name}_{self.atlas}"
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Connectome Comparison: {connectome1_name} vs {connectome2_name} ({self.atlas})', 
                    fontsize=16, fontweight='bold')
        
        # Determine common color scale for both connectomes
        if log_scale and (np.any(true_conn > 0) or np.any(pred_conn > 0)):
            # Find common min/max for consistent scaling
            all_positive_values = np.concatenate([
                true_conn[true_conn > 0].flatten(),
                pred_conn[pred_conn > 0].flatten()
            ])
            if len(all_positive_values) > 0:
                common_vmin = np.min(all_positive_values)
                common_vmax = np.max([np.max(true_conn), np.max(pred_conn)])
                norm = LogNorm(vmin=common_vmin, vmax=common_vmax)
                # Replace zeros with small values for log scale
                true_conn_display = np.where(true_conn == 0, common_vmin * 0.1, true_conn)
                pred_conn_display = np.where(pred_conn == 0, common_vmin * 0.1, pred_conn)
            else:
                norm = None
                true_conn_display = true_conn
                pred_conn_display = pred_conn
        else:
            # Linear scale with common range
            common_vmin = 0
            common_vmax = np.max([np.max(true_conn), np.max(pred_conn)])
            norm = Normalize(vmin=common_vmin, vmax=common_vmax)
            true_conn_display = true_conn
            pred_conn_display = pred_conn
        
        # Plot 1: Ground truth connectome
        im1 = axes[0, 0].imshow(true_conn_display, cmap='jet', norm=norm, aspect='equal', origin='upper')
        axes[0, 0].set_title(f'{connectome1_name} (Ground Truth)')
        axes[0, 0].set_xlabel('Region')
        axes[0, 0].set_ylabel('Region')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.set_label('Connection Strength', rotation=270, labelpad=15)
        
        # Plot 2: Predicted connectome
        im2 = axes[0, 1].imshow(pred_conn_display, cmap='jet', norm=norm, aspect='equal', origin='upper')
        axes[0, 1].set_title(f'{connectome2_name} (Predicted)')
        axes[0, 1].set_xlabel('Region')
        axes[0, 1].set_ylabel('Region')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Connection Strength', rotation=270, labelpad=15)
        
        # Plot 3: Difference matrix
        diff_matrix = pred_conn_display - true_conn_display
        diff_max = np.max(np.abs(diff_matrix))
        im3 = axes[0, 2].imshow(diff_matrix, cmap='RdBu_r', origin='upper', 
                              vmin=-diff_max, vmax=diff_max, aspect='equal')
        axes[0, 2].set_title('Difference (Pred - True)')
        axes[0, 2].set_xlabel('Region')
        axes[0, 2].set_ylabel('Region')
        cbar3 = plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        cbar3.set_label('Difference', rotation=270, labelpad=15)
        
        # Plot 4: Scatter plot
        mask = ~np.eye(true_conn_display.shape[0], dtype=bool)
        true_flat = true_conn_display[mask]
        pred_flat = pred_conn_display[mask]
        
        axes[1, 0].scatter(true_flat, pred_flat, alpha=0.6, s=2, c='blue')
        max_val = max(true_flat.max(), pred_flat.max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2)
        axes[1, 0].set_xlabel('Ground Truth')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title('Scatter Plot')
        
        # Add correlation info
        comparison_name = f"{connectome1_name}_vs_{connectome2_name}"
        
        # Ensure comparison metrics exist - compute them if not already available
        if comparison_name not in self.metrics:
            self._log(f"Computing comparison metrics for plot: {comparison_name}", "info")
            self.compute_comparison_metrics(connectome1_name, connectome2_name, comparison_name)
        
        if comparison_name in self.metrics:
            r = self.metrics[comparison_name]['pearson_r']
            axes[1, 0].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[1, 0].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 5: Error distribution
        error_flat = np.abs(pred_flat - true_flat)
        axes[1, 1].hist(error_flat, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].axvline(np.mean(error_flat), color='red', linestyle='--', 
                          label=f'Mean = {np.mean(error_flat):.3f}')
        axes[1, 1].legend()
        
        # Plot 6: Bland-Altman plot
        mean_vals = (true_flat + pred_flat) / 2
        diff_vals = pred_flat - true_flat
        axes[1, 2].scatter(mean_vals, diff_vals, alpha=0.6, s=2, c='green')
        axes[1, 2].axhline(0, color='red', linestyle='-', alpha=0.8)
        axes[1, 2].axhline(np.mean(diff_vals), color='blue', linestyle='--', alpha=0.8)
        axes[1, 2].axhline(np.mean(diff_vals) + 1.96*np.std(diff_vals), color='gray', linestyle=':', alpha=0.8)
        axes[1, 2].axhline(np.mean(diff_vals) - 1.96*np.std(diff_vals), color='gray', linestyle=':', alpha=0.8)
        axes[1, 2].set_xlabel('Mean of True and Predicted')
        axes[1, 2].set_ylabel('Predicted - True')
        axes[1, 2].set_title('Bland-Altman Plot')
        
        # Add comparison metrics annotation at the bottom of the figure
        if comparison_name in self.metrics:
            metrics = self.metrics[comparison_name]
            
            # Format the metrics string with key comparison scores
            metrics_text = (
                f"Pearson r: {metrics.get('pearson_r', 'N/A'):.3f}  |  "
                f"Spearman r: {metrics.get('spearman_r', 'N/A'):.3f}  |  "
                f"RMSE: {metrics.get('rmse', 'N/A'):.3f}  |  "
                f"MAE: {metrics.get('mae', 'N/A'):.3f}  |  "
                f"Mean LERM: {metrics.get('mean_lerm', 'N/A'):.3f}  |  "
                f"Edge Overlap: {metrics.get('edge_overlap', 'N/A'):.3f}"
            )
            
            # Add text annotation at the bottom of the figure
            fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Adjust layout to leave space for annotation
        
        # Save plot
        plot_path = self.out_path / f"{plot_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self._log(f"Saved comparison plot to {plot_path}")
        return str(plot_path)
    
    def save_results_summary(self, filename: Optional[str] = None) -> str:
        """
        Save comprehensive results summary to CSV files
        
        Args:
            filename: Base filename (auto-generated if None)
            
        Returns:
            Path to saved summary file
        """
        if filename is None:
            filename = f"connectome_analysis_summary_{self.atlas}"
        
        # Save comparison metrics
        if self.metrics:
            metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index')
            metrics_path = self.out_path / f"{filename}_metrics.csv"
            metrics_df.to_csv(metrics_path)
            self._log(f"Saved comparison metrics to {metrics_path}")
        
        # Save network metrics
        if self.network_metrics:
            network_df = pd.DataFrame.from_dict(self.network_metrics, orient='index')
            network_path = self.out_path / f"{filename}_network.csv"
            network_df.to_csv(network_path)
            self._log(f"Saved network metrics to {network_path}")
        
        # Create combined summary
        summary_data = {
            'atlas': self.atlas,
            'atlas_description': self.atlas_description,
            'num_labels': self.num_labels,
            'num_connectomes': len(self.connectomes),
            'num_comparisons': len(self.metrics),
            'connectome_names': list(self.connectomes.keys()),
            'comparison_names': list(self.metrics.keys())
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_path = self.out_path / f"{filename}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        self._log(f"Saved analysis summary to {summary_path}")
        return str(summary_path)
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis"""
        print(f"\n{'='*60}")
        print(f"CONNECTOME ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Atlas: {self.atlas} ({self.atlas_description})")
        print(f"Number of regions: {self.num_labels}")
        print(f"Output directory: {self.out_path}")
        
        print(f"\nConnectomes ({len(self.connectomes)}):")
        for name, connectome in self.connectomes.items():
            total_strength = np.sum(connectome)
            num_connections = np.sum(connectome > 0)
            print(f"  {name}: {connectome.shape} matrix, "
                  f"{num_connections} connections, total strength = {total_strength:.2f}")
        
        if self.metrics:
            print(f"\nComparison Results ({len(self.metrics)}):")
            for comparison_name, metrics in self.metrics.items():
                print(f"  {comparison_name}:")
                print(f"    Pearson r: {metrics.get('pearson_r', 'N/A'):.4f}")
                print(f"    Spearman r: {metrics.get('spearman_r', 'N/A'):.4f}")
                print(f"    RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                print(f"    Mean LERM: {metrics.get('mean_lerm', 'N/A'):.4f}")
        
        if self.network_metrics:
            print(f"\nNetwork Analysis ({len(self.network_metrics)}):")
            for connectome_name, metrics in self.network_metrics.items():
                print(f"  {connectome_name}:")
                print(f"    Density: {metrics.get('density', 'N/A'):.4f}")
                print(f"    Clustering: {metrics.get('average_clustering', 'N/A'):.4f}")
                print(f"    Path length: {metrics.get('average_path_length', 'N/A'):.4f}")
                print(f"    Global efficiency: {metrics.get('global_efficiency', 'N/A'):.4f}")
        
        print(f"{'='*60}\n")


# High-level convenience functions
def analyze_connectomes_from_labels(pred_labels_file: str,
                                  true_labels_file: str,
                                  diffusion_metrics_dir: str,
                                  atlas: str,
                                  out_path: str,
                                  logger = None,
                                  compute_network_advanced: bool = False,
                                  compute_network_centrality: bool = False,
                                  compute_network_community: bool = False) -> ConnectomeAnalyzer:
    """
    High-level function to perform complete connectome analysis from label files
    
    Args:
        pred_labels_file: Path to predicted labels file
        true_labels_file: Path to true labels file  
        diffusion_metrics_dir: Directory containing diffusion metric files
        atlas: Atlas name
        out_path: Output directory
        logger: Logger instance
        compute_network_advanced: Whether to compute advanced network metrics (slow)
        compute_network_centrality: Whether to compute centrality measures (very slow)
        compute_network_community: Whether to compute community detection metrics
        
    Returns:
        ConnectomeAnalyzer instance with results
    """
    analyzer = ConnectomeAnalyzer(atlas=atlas, out_path=out_path, logger=logger)
    
    # Load labels
    with open(pred_labels_file, 'r') as f:
        pred_labels = [int(line.strip()) for line in f if line.strip()]
    
    with open(true_labels_file, 'r') as f:
        true_labels = [int(line.strip()) for line in f if line.strip()]
    
    # Create NoS connectomes
    analyzer.create_connectome_from_labels(pred_labels, connectome_name='pred_nos')
    analyzer.create_connectome_from_labels(true_labels, connectome_name='true_nos')
    
    # Create weighted connectomes for each diffusion metric
    for metric_name, metric_config in DIFFUSION_METRICS.items():
        metric_file = Path(diffusion_metrics_dir) / metric_config['filename']
        
        # Also check for thresholded version if standard file doesn't exist
        if not metric_file.exists():
            # Check for thresholded filename (e.g., mean_fa_per_streamline_thresholded.txt)
            base_name = metric_config['filename'].replace('.txt', '_thresholded.txt')
            metric_file = Path(diffusion_metrics_dir) / base_name
        
        if metric_file.exists():
            try:
                with open(metric_file, 'r') as f:
                    lines = f.readlines()
                
                # Use improved metric loading to handle NaN values correctly
                expected_length = len(pred_labels)
                metric_values_array = analyzer._load_metric_values_improved(metric_file, expected_length, metric_name)
                
                # Convert to list for compatibility with existing code
                metric_values = metric_values_array.tolist()
                
                if len(metric_values) > 0:
                    # Ensure same length as labels (should now be consistent)
                    min_len = min(len(pred_labels), len(metric_values))
                    if min_len != len(pred_labels):
                        analyzer._log(f"Metric {metric_name}: using {min_len} of {len(pred_labels)} expected values", "warning")
                    
                    labels_subset = pred_labels[:min_len]
                    values_subset = metric_values[:min_len]
                    
                    # SIFT2 weights should be summed, other metrics should be averaged
                    aggregation_method = 'sum' if metric_name == 'sift2' else 'mean'
                    
                    analyzer.create_connectome_from_labels(
                        labels_subset, weights=values_subset, 
                        connectome_name=f'pred_{metric_name}',
                        aggregation_method=aggregation_method
                    )
                    analyzer.create_connectome_from_labels(
                        true_labels[:min_len], weights=values_subset,
                        connectome_name=f'true_{metric_name}',
                        aggregation_method=aggregation_method
                    )
                
            except Exception as e:
                analyzer._log(f"Error loading {metric_name} values: {e}", "warning")
    
    # Save all connectomes
    for connectome_name in analyzer.connectomes:
        analyzer.save_connectome(connectome_name)
    
    # Perform comparisons
    comparison_types = ['nos'] + list(DIFFUSION_METRICS.keys())
    
    for comp_type in comparison_types:
        pred_name = f'pred_{comp_type}'
        true_name = f'true_{comp_type}'
        
        if pred_name in analyzer.connectomes and true_name in analyzer.connectomes:
            # Compute comparison metrics - use consistent naming
            comparison_name = f"{true_name}_vs_{pred_name}"
            analyzer.compute_comparison_metrics(true_name, pred_name, comparison_name)
            
            # Create comparison plot - will use the same naming convention
            analyzer.create_comparison_plot(true_name, pred_name, f'{comp_type}_comparison')
            
            # Compute network metrics
            analyzer.compute_network_metrics(pred_name, 
                                           compute_advanced=compute_network_advanced,
                                           compute_centrality=compute_network_centrality,
                                           compute_community=compute_network_community)
            analyzer.compute_network_metrics(true_name,
                                           compute_advanced=compute_network_advanced,
                                           compute_centrality=compute_network_centrality,
                                           compute_community=compute_network_community)
    
    # Save results
    analyzer.save_results_summary()
    analyzer.print_summary()
    
    return analyzer