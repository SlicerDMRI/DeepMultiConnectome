#!/usr/bin/env python3
"""
Population comparison helper functions for intra_inter_subject_analysis.py

This module provides functions to:
1. Compute population average connectomes from all subjects
2. Compare individual subjects against population averages
3. Create 3-way comparison boxplots (pred_vs_true, pred_vs_pop, true_vs_pop)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.linalg import logm, norm

from .analysis_metrics import apply_zero_mask, compute_correlation, compute_lerm


def compute_population_average_connectomes(subject_list: List[str], output_path: Path, 
                                          atlases: List[str], connectome_types: List[str],
                                          get_subject_output_dir_func, file_suffix: str = "") -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute population average connectomes.
    If a saved average file exists in output_path, load it.
    Otherwise, compute from available subjects in subject_list and SAVE it.
    
    Args:
        subject_list: List of subject IDs to use for computation
        output_path: Directory to save/load population averages (e.g. results_dir)
        atlases: List of atlas names
        connectome_types: List of connectome types
        get_subject_output_dir_func: Function to get subject output directory
        file_suffix: Suffix for connectome files (e.g. '_20mm')
    """
    
    print(f"\nChecking population average connectomes (suffix='{file_suffix}')...")
    
    population_averages = {}
    
    # Ensure output directory for averages exists
    pop_avg_dir = output_path / "population_averages"
    pop_avg_dir.mkdir(parents=True, exist_ok=True)
    
    for atlas in atlases:
        population_averages[atlas] = {}
        
        for connectome_type in connectome_types:
            # 1. Try Loading from Disk
            avg_filename = f"pop_average_{atlas}_{connectome_type}{file_suffix}.csv"
            avg_file = pop_avg_dir / avg_filename
            
            loaded_valid = False
            if avg_file.exists():
                try:
                    df = pd.read_csv(avg_file, header=None)
                    population_averages[atlas][connectome_type] = df.values
                    # Quietly loaded
                    loaded_valid = True
                except Exception as e:
                    print(f"    Failed to load cached average: {e}")
            
            if loaded_valid:
                continue

            # 2. Compute from Scratch (if not loaded)
            print(f"  Computing population average for {atlas} - {connectome_type} from {len(subject_list)} subjects...")
            connectomes = []
            
            for subject_id in subject_list:
                output_dir = get_subject_output_dir_func(subject_id, atlas)
                true_file = output_dir / f"connectome_true_{connectome_type}_{atlas}{file_suffix}.csv"
                
                if true_file.exists():
                    try:
                        true_matrix = pd.read_csv(true_file, header=None).values
                        if true_matrix.size > 0:
                            connectomes.append(true_matrix)
                    except Exception as e:
                        # print(f"    Warning: Could not load {subject_id}: {e}")
                        pass
            
            if len(connectomes) == 0:
                print(f"    No valid connectomes found for {atlas} - {connectome_type}")
                continue
            
            # Compute average
            avg_connectome = np.mean(connectomes, axis=0)
            population_averages[atlas][connectome_type] = avg_connectome
            
            print(f"    Computed average from {len(connectomes)} subjects. Saving to {avg_file}...")
            
            # 3. Save to Disk
            try:
                pd.DataFrame(avg_connectome).to_csv(avg_file, header=False, index=False)
            except Exception as e:
                print(f"    Warning: Failed to save population average: {e}")
    
    return population_averages


def compute_population_comparisons(all_metrics: Dict, population_averages: Dict,
                                   atlases: List[str], connectome_types: List[str],
                                   get_subject_output_dir_func, mask_zeros: bool = True,
                                   no_diagonal: bool = False, file_suffix: str = "") -> Dict:
    """
    Compare each subject against population averages and compute 3-way metrics
    
    Args:
        all_metrics: Dictionary of all subject metrics (will be updated)
        population_averages: Population average connectomes
        atlases: List of atlas names
        connectome_types: List of connectome types
        get_subject_output_dir_func: Function to get subject output directory
        mask_zeros: Whether to mask zeros for diffusion metrics
        no_diagonal: Whether to exclude diagonal
        file_suffix: Suffix for connectome files
        
    Returns:
        Updated all_metrics dictionary with population comparison metrics
    """
    
    print("\nComparing subjects against population averages...")
    
    subject_count = 0
    for subject_id in all_metrics.keys():
        subject_count += 1
        if subject_count % 50 == 0:  # Progress indicator
            print(f"  Processing subject {subject_count}/{len(all_metrics)}...")
        
        for atlas in atlases:
            if atlas not in population_averages:
                continue
            
            for connectome_type in connectome_types:
                if connectome_type not in population_averages[atlas]:
                    continue
                
                # Skip if no metrics for this combination
                if (atlas not in all_metrics[subject_id] or 
                    connectome_type not in all_metrics[subject_id][atlas]):
                    continue
                
                # Get population average
                pop_avg = population_averages[atlas][connectome_type]
                
                # Load subject's connectomes
                output_dir = get_subject_output_dir_func(subject_id, atlas)
                pred_file = output_dir / f"connectome_pred_{connectome_type}_{atlas}{file_suffix}.csv"
                true_file = output_dir / f"connectome_true_{connectome_type}_{atlas}{file_suffix}.csv"
                
                if not (pred_file.exists() and true_file.exists()):
                    continue
                
                try:
                    pred_matrix = pd.read_csv(pred_file, header=None).values
                    true_matrix = pd.read_csv(true_file, header=None).values
                    
                    if pred_matrix.size == 0 or true_matrix.size == 0: continue

                    
                    # Compute pred vs pop
                    pred_vs_pop_r = compute_correlation(
                        pop_avg, pred_matrix, include_diagonal=not no_diagonal,
                        filter_zeros=mask_zeros
                    )
                    pred_vs_pop_lerm = compute_lerm(pop_avg, pred_matrix, use_matrix_log=True)
                    
                    # Compute true vs pop (natural variability baseline)
                    true_vs_pop_r = compute_correlation(
                        pop_avg, true_matrix, include_diagonal=not no_diagonal,
                        filter_zeros=mask_zeros
                    )
                    true_vs_pop_lerm = compute_lerm(pop_avg, true_matrix, use_matrix_log=True)
                    
                    # Store in metrics
                    all_metrics[subject_id][atlas][connectome_type]['pred_vs_pop_r'] = pred_vs_pop_r
                    all_metrics[subject_id][atlas][connectome_type]['pred_vs_pop_lerm'] = pred_vs_pop_lerm
                    all_metrics[subject_id][atlas][connectome_type]['true_vs_pop_r'] = true_vs_pop_r
                    all_metrics[subject_id][atlas][connectome_type]['true_vs_pop_lerm'] = true_vs_pop_lerm
                    
                except Exception as e:
                    print(f"    Error processing {subject_id} {atlas} {connectome_type}: {e}")
                    continue
    
    print("  Population comparisons completed")
    return all_metrics


def create_three_way_boxplot(all_metrics: Dict, results_dir: Path, 
                             atlases: List[str], connectome_types: List[str] = ['nos', 'fa', 'sift2']):
    """
    Create 3-way comparison boxplots showing:
    - pred_vs_true (individual subject accuracy)
    - pred_vs_pop (prediction vs population average)
    - true_vs_pop (natural population variability baseline)
    
    Creates plots for NOS, FA, and SIFT2 connectome types
    
    Args:
        all_metrics: Dictionary with all subject metrics
        results_dir: Directory to save plots
        atlases: List of atlas names
        connectome_types: Connectome types to plot (default: nos, fa, sift2)
    """
    
    print("\nCreating 3-way comparison boxplots...")
    
    plots_dir = results_dir / "plots" / "population_comparison"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for all three connectome types
    for connectome_type in connectome_types:
        print(f"  Creating plot for {connectome_type.upper()}...")
        
        # Prepare data for correlation
        data_r = []
        data_lerm = []
        
        for atlas in atlases:
            atlas_label = "84 ROIs" if "aparc+aseg" == atlas else "164 ROIs"
            
            for subject_id in all_metrics.keys():
                if (atlas not in all_metrics[subject_id] or 
                    connectome_type not in all_metrics[subject_id][atlas]):
                    continue
                
                metrics = all_metrics[subject_id][atlas][connectome_type]
                
                # Get all three comparison types
                pred_vs_true_r = metrics.get('intra_r', np.nan)
                pred_vs_pop_r = metrics.get('pred_vs_pop_r', np.nan)
                true_vs_pop_r = metrics.get('true_vs_pop_r', np.nan)
                
                pred_vs_true_lerm = metrics.get('intra_lerm', np.nan)
                pred_vs_pop_lerm = metrics.get('pred_vs_pop_lerm', np.nan)
                true_vs_pop_lerm = metrics.get('true_vs_pop_lerm', np.nan)
                
                # Add to data if valid
                if not np.isnan(pred_vs_true_r):
                    data_r.append({
                        'Atlas': atlas_label,
                        'Comparison': 'Pred vs True',
                        'Pearson r': pred_vs_true_r
                    })
                
                if not np.isnan(pred_vs_pop_r):
                    data_r.append({
                        'Atlas': atlas_label,
                        'Comparison': 'Pred vs Population',
                        'Pearson r': pred_vs_pop_r
                    })
                
                if not np.isnan(true_vs_pop_r):
                    data_r.append({
                        'Atlas': atlas_label,
                        'Comparison': 'True vs Population',
                        'Pearson r': true_vs_pop_r
                    })
                
                # LERM data
                if not np.isnan(pred_vs_true_lerm):
                    data_lerm.append({
                        'Atlas': atlas_label,
                        'Comparison': 'Pred vs True',
                        'LERM': pred_vs_true_lerm
                    })
                
                if not np.isnan(pred_vs_pop_lerm):
                    data_lerm.append({
                        'Atlas': atlas_label,
                        'Comparison': 'Pred vs Population',
                        'LERM': pred_vs_pop_lerm
                    })
                
                if not np.isnan(true_vs_pop_lerm):
                    data_lerm.append({
                        'Atlas': atlas_label,
                        'Comparison': 'True vs Population',
                        'LERM': true_vs_pop_lerm
                    })
        
        if len(data_r) == 0:
            print(f"    No data available for {connectome_type}, skipping")
            continue
        
        # Create DataFrame
        df_r = pd.DataFrame(data_r)
        df_lerm = pd.DataFrame(data_lerm)
        
        # Create figure with 1x2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Pearson correlation
        sns.boxplot(data=df_r, x='Atlas', y='Pearson r', hue='Comparison', ax=axes[0])
        axes[0].set_title(f'{connectome_type.upper()} - Pearson Correlation', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Pearson r', fontsize=12)
        axes[0].set_xlabel('Atlas', fontsize=12)
        axes[0].legend(title='Comparison Type', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: LERM
        sns.boxplot(data=df_lerm, x='Atlas', y='LERM', hue='Comparison', ax=axes[1])
        axes[1].set_title(f'{connectome_type.upper()} - Log-Euclidean Distance (LERM)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('LERM', fontsize=12)
        axes[1].set_xlabel('Atlas', fontsize=12)
        axes[1].legend(title='Comparison Type', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = plots_dir / f"three_way_comparison_{connectome_type}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {plot_file}")
    
    print(f"  Three-way comparison plots saved to {plots_dir}")
