#!/usr/bin/env python3
"""
Connectome Mismatch Analysis Script

This script analyzes mismatches between true and predicted connectomes,
specifically focusing on cases where one connectome has zero values while
the other doesn't. It tests the hypothesis that these mismatches occur
primarily in connections with low streamline counts.

Usage:
    python analyze_connectome_mismatches.py --subject 100206 --atlas aparc+aseg
    python analyze_connectome_mismatches.py --subject 100206 --atlas aparc.a2009s+aseg --metric fa
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.unified_connectome import ConnectomeAnalyzer, analyze_connectomes_from_labels


class ConnectomeMismatchAnalyzer:
    """Analyze mismatches between true and predicted connectomes"""
    
    def __init__(self, subject_id: str = "100206", atlas: str = "aparc+aseg"):
        self.subject_id = subject_id
        self.atlas = atlas
        
        # Set up paths
        self.base_path = Path("/media/volume/MV_HCP")
        self.subject_path = self.base_path / "HCP_MRtrix" / self.subject_id
        self.diffusion_dir = self.subject_path / "dMRI"
        self.output_dir = Path(f"/tmp/mismatch_analysis_{subject_id}_{atlas}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Analyzing connectome mismatches for subject {subject_id}, atlas {atlas}")
        print(f"Output directory: {self.output_dir}")
    
    def load_connectomes(self, metric: str = "fa") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load true, predicted, and NOS (number of streamlines) connectomes
        
        Args:
            metric: Diffusion metric to analyze ('fa', 'md', 'ad', 'rd', 'sift2')
            
        Returns:
            true_connectome, pred_connectome, nos_connectome
        """
        print(f"\nLoading {metric.upper()} connectomes...")
        
        # Define file paths
        pred_labels_file = self.subject_path / "TractCloud" / f"predictions_{self.atlas}_symmetric.txt"
        true_labels_file = self.subject_path / "output" / f"labels_10M_{self.atlas}_symmetric.txt"
        
        if not pred_labels_file.exists():
            raise FileNotFoundError(f"Predicted labels file not found: {pred_labels_file}")
        if not true_labels_file.exists():
            raise FileNotFoundError(f"True labels file not found: {true_labels_file}")
        
        # Run analysis to generate connectomes
        analyzer = analyze_connectomes_from_labels(
            pred_labels_file=str(pred_labels_file),
            true_labels_file=str(true_labels_file),
            diffusion_metrics_dir=str(self.diffusion_dir),
            atlas=self.atlas,
            out_path=str(self.output_dir),
            compute_network_advanced=False,
            compute_network_centrality=False,
            compute_network_community=False
        )
        
        if analyzer is None:
            raise RuntimeError("Failed to create connectome analyzer")
        
        # Extract connectomes
        true_connectome = analyzer.connectomes[f'true_{metric}']
        pred_connectome = analyzer.connectomes[f'pred_{metric}']
        nos_true = analyzer.connectomes['true_nos']
        nos_pred = analyzer.connectomes['pred_nos']
        
        print(f"Loaded connectomes:")
        print(f"  True {metric}: {true_connectome.shape}, {np.count_nonzero(true_connectome)} connections")
        print(f"  Pred {metric}: {pred_connectome.shape}, {np.count_nonzero(pred_connectome)} connections")
        print(f"  True NOS: {nos_true.shape}, {np.count_nonzero(nos_true)} connections")
        print(f"  Pred NOS: {nos_pred.shape}, {np.count_nonzero(nos_pred)} connections")
        
        return true_connectome, pred_connectome, nos_true, nos_pred, analyzer
    
    def analyze_mismatches(self, true_connectome: np.ndarray, pred_connectome: np.ndarray,
                          nos_true: np.ndarray, nos_pred: np.ndarray, metric: str = "fa") -> Dict:
        """
        Analyze mismatches between true and predicted connectomes
        
        Args:
            true_connectome: True connectome matrix
            pred_connectome: Predicted connectome matrix  
            nos_true: True number of streamlines matrix
            nos_pred: Predicted number of streamlines matrix
            metric: Metric name for labeling
            
        Returns:
            Dictionary with mismatch analysis results
        """
        print(f"\nAnalyzing {metric.upper()} connectome mismatches...")
        
        # Create masks for different mismatch types
        true_zero = (true_connectome == 0)
        pred_zero = (pred_connectome == 0)
        
        # Type 1: True=0, Pred>0 (horizontal line in scatter plot)
        horizontal_mismatch = true_zero & ~pred_zero
        
        # Type 2: True>0, Pred=0 (vertical line in scatter plot)  
        vertical_mismatch = ~true_zero & pred_zero
        
        # Type 3: Both non-zero (normal data points)
        both_nonzero = ~true_zero & ~pred_zero
        
        # Type 4: Both zero (expected zeros)
        both_zero = true_zero & pred_zero
        
        print(f"Mismatch types:")
        print(f"  Horizontal (True=0, Pred>0): {np.sum(horizontal_mismatch)} connections")
        print(f"  Vertical (True>0, Pred=0): {np.sum(vertical_mismatch)} connections")
        print(f"  Both non-zero: {np.sum(both_nonzero)} connections")
        print(f"  Both zero: {np.sum(both_zero)} connections")
        
        # Analyze streamline counts for each mismatch type
        results = {
            'metric': metric,
            'horizontal_mismatch': {
                'count': np.sum(horizontal_mismatch),
                'true_nos': nos_true[horizontal_mismatch],
                'pred_nos': nos_pred[horizontal_mismatch],
                'pred_values': pred_connectome[horizontal_mismatch]
            },
            'vertical_mismatch': {
                'count': np.sum(vertical_mismatch),
                'true_nos': nos_true[vertical_mismatch],
                'pred_nos': nos_pred[vertical_mismatch],
                'true_values': true_connectome[vertical_mismatch]
            },
            'both_nonzero': {
                'count': np.sum(both_nonzero),
                'true_nos': nos_true[both_nonzero],
                'pred_nos': nos_pred[both_nonzero],
                'true_values': true_connectome[both_nonzero],
                'pred_values': pred_connectome[both_nonzero]
            },
            'both_zero': {
                'count': np.sum(both_zero),
                'true_nos': nos_true[both_zero],
                'pred_nos': nos_pred[both_zero]
            }
        }
        
        return results
    
    def create_mismatch_statistics(self, results: Dict) -> pd.DataFrame:
        """Create detailed statistics for each mismatch type"""
        
        stats_data = []
        
        for mismatch_type, data in results.items():
            if mismatch_type == 'metric':
                continue
                
            if data['count'] > 0:
                true_nos_stats = {
                    'mismatch_type': mismatch_type,
                    'streamline_type': 'true_nos',
                    'count': data['count'],
                    'mean': np.mean(data['true_nos']),
                    'median': np.median(data['true_nos']),
                    'std': np.std(data['true_nos']),
                    'min': np.min(data['true_nos']),
                    'max': np.max(data['true_nos']),
                    'q25': np.percentile(data['true_nos'], 25),
                    'q75': np.percentile(data['true_nos'], 75),
                    'below_1': np.sum(data['true_nos'] < 1),
                    'below_5': np.sum(data['true_nos'] < 5),
                    'below_10': np.sum(data['true_nos'] < 10),
                    'below_50': np.sum(data['true_nos'] < 50),
                    'below_100': np.sum(data['true_nos'] < 100)
                }
                
                pred_nos_stats = {
                    'mismatch_type': mismatch_type,
                    'streamline_type': 'pred_nos',
                    'count': data['count'],
                    'mean': np.mean(data['pred_nos']),
                    'median': np.median(data['pred_nos']),
                    'std': np.std(data['pred_nos']),
                    'min': np.min(data['pred_nos']),
                    'max': np.max(data['pred_nos']),
                    'q25': np.percentile(data['pred_nos'], 25),
                    'q75': np.percentile(data['pred_nos'], 75),
                    'below_1': np.sum(data['pred_nos'] < 1),
                    'below_5': np.sum(data['pred_nos'] < 5),
                    'below_10': np.sum(data['pred_nos'] < 10),
                    'below_50': np.sum(data['pred_nos'] < 50),
                    'below_100': np.sum(data['pred_nos'] < 100)
                }
                
                stats_data.extend([true_nos_stats, pred_nos_stats])
        
        return pd.DataFrame(stats_data)
    
    def create_visualizations(self, results: Dict, metric: str = "fa"):
        """Create comprehensive visualizations of the mismatch analysis"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Scatter plot with mismatch highlighting
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Plot all data points
        both_nonzero = results['both_nonzero']
        if both_nonzero['count'] > 0:
            ax1.scatter(both_nonzero['true_values'], both_nonzero['pred_values'], 
                       alpha=0.6, s=20, c='blue', label=f"Both non-zero ({both_nonzero['count']})")
        
        # Highlight mismatches
        horizontal = results['horizontal_mismatch']
        if horizontal['count'] > 0:
            ax1.scatter(np.zeros(horizontal['count']), horizontal['pred_values'],
                       alpha=0.8, s=30, c='red', marker='s', 
                       label=f"True=0, Pred>0 ({horizontal['count']})")
        
        vertical = results['vertical_mismatch']
        if vertical['count'] > 0:
            ax1.scatter(vertical['true_values'], np.zeros(vertical['count']),
                       alpha=0.8, s=30, c='orange', marker='^',
                       label=f"True>0, Pred=0 ({vertical['count']})")
        
        ax1.set_xlabel(f'True {metric.upper()}')
        ax1.set_ylabel(f'Predicted {metric.upper()}')
        ax1.set_title(f'{metric.upper()} Connectome: True vs Predicted with Mismatches')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line
        max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect match')
        
        # 2. Streamline count distributions for horizontal mismatches
        ax2 = fig.add_subplot(gs[0, 2])
        if horizontal['count'] > 0:
            ax2.hist(horizontal['true_nos'], bins=50, alpha=0.7, color='red', 
                    label=f'True NOS (n={horizontal["count"]})')
            ax2.hist(horizontal['pred_nos'], bins=50, alpha=0.7, color='darkred',
                    label=f'Pred NOS (n={horizontal["count"]})')
            ax2.set_xlabel('Number of Streamlines')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Horizontal Mismatches\n(True=0, Pred>0)')
            ax2.legend()
            ax2.set_yscale('log')
        
        # 3. Streamline count distributions for vertical mismatches  
        ax3 = fig.add_subplot(gs[0, 3])
        if vertical['count'] > 0:
            ax3.hist(vertical['true_nos'], bins=50, alpha=0.7, color='orange',
                    label=f'True NOS (n={vertical["count"]})')
            ax3.hist(vertical['pred_nos'], bins=50, alpha=0.7, color='darkorange',
                    label=f'Pred NOS (n={vertical["count"]})')
            ax3.set_xlabel('Number of Streamlines')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Vertical Mismatches\n(True>0, Pred=0)')
            ax3.legend()
            ax3.set_yscale('log')
        
        # 4. Cumulative distribution of streamline counts
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Plot CDFs for different mismatch types
        if horizontal['count'] > 0:
            sorted_h_true = np.sort(horizontal['true_nos'])
            sorted_h_pred = np.sort(horizontal['pred_nos'])
            ax4.plot(sorted_h_true, np.arange(1, len(sorted_h_true)+1)/len(sorted_h_true),
                    'r-', linewidth=2, label=f'Horizontal True NOS')
            ax4.plot(sorted_h_pred, np.arange(1, len(sorted_h_pred)+1)/len(sorted_h_pred),
                    'r--', linewidth=2, label=f'Horizontal Pred NOS')
        
        if vertical['count'] > 0:
            sorted_v_true = np.sort(vertical['true_nos'])
            sorted_v_pred = np.sort(vertical['pred_nos'])
            ax4.plot(sorted_v_true, np.arange(1, len(sorted_v_true)+1)/len(sorted_v_true),
                    'orange', linewidth=2, label=f'Vertical True NOS')
            ax4.plot(sorted_v_pred, np.arange(1, len(sorted_v_pred)+1)/len(sorted_v_pred),
                    'orange', linestyle='--', linewidth=2, label=f'Vertical Pred NOS')
        
        if both_nonzero['count'] > 0:
            sorted_b_true = np.sort(both_nonzero['true_nos'])
            sorted_b_pred = np.sort(both_nonzero['pred_nos'])
            # Sample for performance if too many points
            if len(sorted_b_true) > 10000:
                idx = np.linspace(0, len(sorted_b_true)-1, 5000, dtype=int)
                sorted_b_true = sorted_b_true[idx]
                sorted_b_pred = sorted_b_pred[idx]
            ax4.plot(sorted_b_true, np.arange(1, len(sorted_b_true)+1)/len(sorted_b_true),
                    'b-', linewidth=1, alpha=0.7, label=f'Both Non-zero True NOS')
            ax4.plot(sorted_b_pred, np.arange(1, len(sorted_b_pred)+1)/len(sorted_b_pred),
                    'b--', linewidth=1, alpha=0.7, label=f'Both Non-zero Pred NOS')
        
        ax4.set_xlabel('Number of Streamlines')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution of Streamline Counts')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        # Add vertical lines for common thresholds
        for threshold in [1, 5, 10, 50, 100]:
            ax4.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
            ax4.text(threshold, 0.9, f'{threshold}', rotation=90, alpha=0.7)
        
        # 5. Box plots comparing streamline counts
        ax5 = fig.add_subplot(gs[1, 2:])
        
        box_data = []
        box_labels = []
        
        if horizontal['count'] > 0:
            box_data.extend([horizontal['true_nos'], horizontal['pred_nos']])
            box_labels.extend(['Horiz True NOS', 'Horiz Pred NOS'])
        
        if vertical['count'] > 0:
            box_data.extend([vertical['true_nos'], vertical['pred_nos']])
            box_labels.extend(['Vert True NOS', 'Vert Pred NOS'])
        
        if both_nonzero['count'] > 0:
            # Sample for performance
            sample_size = min(5000, both_nonzero['count'])
            idx = np.random.choice(both_nonzero['count'], sample_size, replace=False)
            box_data.extend([both_nonzero['true_nos'][idx], both_nonzero['pred_nos'][idx]])
            box_labels.extend(['Both True NOS', 'Both Pred NOS'])
        
        if box_data:
            bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors = ['red', 'darkred', 'orange', 'darkorange', 'blue', 'darkblue']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax5.set_ylabel('Number of Streamlines')
            ax5.set_title('Streamline Count Distributions by Mismatch Type')
            ax5.set_yscale('log')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Threshold analysis
        ax6 = fig.add_subplot(gs[2, :2])
        
        thresholds = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        h_below_true = [np.sum(horizontal['true_nos'] < t) / max(horizontal['count'], 1) * 100 
                       if horizontal['count'] > 0 else 0 for t in thresholds]
        h_below_pred = [np.sum(horizontal['pred_nos'] < t) / max(horizontal['count'], 1) * 100 
                       if horizontal['count'] > 0 else 0 for t in thresholds]
        v_below_true = [np.sum(vertical['true_nos'] < t) / max(vertical['count'], 1) * 100 
                       if vertical['count'] > 0 else 0 for t in thresholds]
        v_below_pred = [np.sum(vertical['pred_nos'] < t) / max(vertical['count'], 1) * 100 
                       if vertical['count'] > 0 else 0 for t in thresholds]
        
        ax6.plot(thresholds, h_below_true, 'r-o', label='Horizontal True NOS')
        ax6.plot(thresholds, h_below_pred, 'r--s', label='Horizontal Pred NOS')
        ax6.plot(thresholds, v_below_true, 'orange', marker='o', label='Vertical True NOS')
        ax6.plot(thresholds, v_below_pred, 'orange', linestyle='--', marker='s', label='Vertical Pred NOS')
        
        ax6.set_xlabel('Streamline Count Threshold')
        ax6.set_ylabel('Percentage Below Threshold')
        ax6.set_title('Percentage of Mismatches Below Streamline Thresholds')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')
        
        # 7. Mismatch value distributions
        ax7 = fig.add_subplot(gs[2, 2])
        if horizontal['count'] > 0:
            ax7.hist(horizontal['pred_values'], bins=50, alpha=0.7, color='red',
                    label=f'Pred {metric.upper()} values\n(True=0, n={horizontal["count"]})')
            ax7.set_xlabel(f'Predicted {metric.upper()} Value')
            ax7.set_ylabel('Frequency')
            ax7.set_title('Distribution of Non-zero\nPredicted Values\n(when True=0)')
            ax7.set_yscale('log')
        
        ax8 = fig.add_subplot(gs[2, 3])
        if vertical['count'] > 0:
            ax8.hist(vertical['true_values'], bins=50, alpha=0.7, color='orange',
                    label=f'True {metric.upper()} values\n(Pred=0, n={vertical["count"]})')
            ax8.set_xlabel(f'True {metric.upper()} Value')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Distribution of Non-zero\nTrue Values\n(when Pred=0)')
            ax8.set_yscale('log')
        
        # 8. Summary statistics table
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table
        summary_data = []
        for mismatch_type in ['horizontal_mismatch', 'vertical_mismatch', 'both_nonzero']:
            data = results[mismatch_type]
            if data['count'] > 0:
                summary_data.append([
                    mismatch_type.replace('_', ' ').title(),
                    data['count'],
                    f"{np.mean(data['true_nos']):.1f}",
                    f"{np.median(data['true_nos']):.1f}",
                    f"{np.sum(data['true_nos'] < 10) / data['count'] * 100:.1f}%",
                    f"{np.mean(data['pred_nos']):.1f}",
                    f"{np.median(data['pred_nos']):.1f}",
                    f"{np.sum(data['pred_nos'] < 10) / data['count'] * 100:.1f}%"
                ])
        
        if summary_data:
            table = ax9.table(cellText=summary_data,
                             colLabels=['Mismatch Type', 'Count', 'True NOS Mean', 'True NOS Median', 
                                       'True <10 Streams', 'Pred NOS Mean', 'Pred NOS Median', 'Pred <10 Streams'],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        plt.suptitle(f'{metric.upper()} Connectome Mismatch Analysis: Subject {self.subject_id}, Atlas {self.atlas}', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        plot_path = self.output_dir / f"mismatch_analysis_{metric}_{self.atlas}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_path}")
    
    def run_analysis(self, metric: str = "fa") -> Dict:
        """Run complete mismatch analysis"""
        
        print(f"\n{'='*80}")
        print(f"CONNECTOME MISMATCH ANALYSIS")
        print(f"Subject: {self.subject_id}")
        print(f"Atlas: {self.atlas}")
        print(f"Metric: {metric.upper()}")
        print(f"{'='*80}")
        
        # Load connectomes
        true_connectome, pred_connectome, nos_true, nos_pred, analyzer = self.load_connectomes(metric)
        
        # Analyze mismatches
        results = self.analyze_mismatches(true_connectome, pred_connectome, nos_true, nos_pred, metric)
        
        # Create statistics
        stats_df = self.create_mismatch_statistics(results)
        
        # Save statistics
        stats_path = self.output_dir / f"mismatch_statistics_{metric}_{self.atlas}.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\nStatistics saved to: {stats_path}")
        
        # Print key findings
        self.print_findings(results, metric)
        
        # Create visualizations
        self.create_visualizations(results, metric)
        
        return results
    
    def print_findings(self, results: Dict, metric: str):
        """Print key findings from the analysis"""
        
        print(f"\n{'='*60}")
        print(f"KEY FINDINGS FOR {metric.upper()} CONNECTOME MISMATCHES")
        print(f"{'='*60}")
        
        horizontal = results['horizontal_mismatch']
        vertical = results['vertical_mismatch']
        both_nonzero = results['both_nonzero']
        
        total_connections = horizontal['count'] + vertical['count'] + both_nonzero['count']
        
        print(f"\n1. MISMATCH PREVALENCE:")
        print(f"   Total connections analyzed: {total_connections}")
        print(f"   Horizontal mismatches (True=0, Pred>0): {horizontal['count']} ({horizontal['count']/total_connections*100:.1f}%)")
        print(f"   Vertical mismatches (True>0, Pred=0): {vertical['count']} ({vertical['count']/total_connections*100:.1f}%)")
        print(f"   Both non-zero (normal): {both_nonzero['count']} ({both_nonzero['count']/total_connections*100:.1f}%)")
        
        print(f"\n2. STREAMLINE COUNT HYPOTHESIS TEST:")
        
        if horizontal['count'] > 0:
            h_low_true = np.sum(horizontal['true_nos'] < 10) / horizontal['count'] * 100
            h_low_pred = np.sum(horizontal['pred_nos'] < 10) / horizontal['count'] * 100
            print(f"   Horizontal mismatches:")
            print(f"     True NOS <10 streamlines: {h_low_true:.1f}%")
            print(f"     Pred NOS <10 streamlines: {h_low_pred:.1f}%")
            print(f"     Mean True NOS: {np.mean(horizontal['true_nos']):.1f}")
            print(f"     Mean Pred NOS: {np.mean(horizontal['pred_nos']):.1f}")
        
        if vertical['count'] > 0:
            v_low_true = np.sum(vertical['true_nos'] < 10) / vertical['count'] * 100
            v_low_pred = np.sum(vertical['pred_nos'] < 10) / vertical['count'] * 100
            print(f"   Vertical mismatches:")
            print(f"     True NOS <10 streamlines: {v_low_true:.1f}%")
            print(f"     Pred NOS <10 streamlines: {v_low_pred:.1f}%")
            print(f"     Mean True NOS: {np.mean(vertical['true_nos']):.1f}")
            print(f"     Mean Pred NOS: {np.mean(vertical['pred_nos']):.1f}")
        
        if both_nonzero['count'] > 0:
            b_low_true = np.sum(both_nonzero['true_nos'] < 10) / both_nonzero['count'] * 100
            b_low_pred = np.sum(both_nonzero['pred_nos'] < 10) / both_nonzero['count'] * 100
            print(f"   Both non-zero (reference):")
            print(f"     True NOS <10 streamlines: {b_low_true:.1f}%")
            print(f"     Pred NOS <10 streamlines: {b_low_pred:.1f}%")
            print(f"     Mean True NOS: {np.mean(both_nonzero['true_nos']):.1f}")
            print(f"     Mean Pred NOS: {np.mean(both_nonzero['pred_nos']):.1f}")
        
        print(f"\n3. HYPOTHESIS EVALUATION:")
        if horizontal['count'] > 0 or vertical['count'] > 0:
            if horizontal['count'] > 0:
                h_hypothesis = h_low_true > 50 or h_low_pred > 50
                print(f"   Horizontal mismatches have low streamline counts: {h_hypothesis}")
            if vertical['count'] > 0:
                v_hypothesis = v_low_true > 50 or v_low_pred > 50
                print(f"   Vertical mismatches have low streamline counts: {v_hypothesis}")
            
            if both_nonzero['count'] > 0:
                print(f"   For comparison, normal connections with <10 streamlines: {b_low_true:.1f}% (true), {b_low_pred:.1f}% (pred)")
        else:
            print("   No mismatches found to evaluate hypothesis.")
        
        print(f"\n{'='*60}")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Analyze connectome mismatches')
    parser.add_argument('--subject', '-s', default='100206', help='Subject ID')
    parser.add_argument('--atlas', '-a', default='aparc+aseg', 
                       choices=['aparc+aseg', 'aparc.a2009s+aseg'],
                       help='Atlas to use')
    parser.add_argument('--metric', '-m', default='fa',
                       choices=['fa', 'md', 'ad', 'rd', 'sift2'],
                       help='Diffusion metric to analyze')
    parser.add_argument('--all-metrics', action='store_true',
                       help='Analyze all available metrics')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ConnectomeMismatchAnalyzer(subject_id=args.subject, atlas=args.atlas)
    
    # Run analysis
    if args.all_metrics:
        metrics = ['fa', 'md', 'ad', 'rd', 'sift2']
        all_results = {}
        for metric in metrics:
            try:
                print(f"\n{'#'*80}")
                print(f"ANALYZING {metric.upper()} METRIC")
                print(f"{'#'*80}")
                results = analyzer.run_analysis(metric)
                all_results[metric] = results
            except Exception as e:
                print(f"Error analyzing {metric}: {e}")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY ACROSS ALL METRICS")
        print(f"{'='*80}")
        
        for metric, results in all_results.items():
            h_count = results['horizontal_mismatch']['count']
            v_count = results['vertical_mismatch']['count']
            b_count = results['both_nonzero']['count']
            total = h_count + v_count + b_count
            print(f"{metric.upper()}: {h_count} horizontal ({h_count/total*100:.1f}%), "
                  f"{v_count} vertical ({v_count/total*100:.1f}%), "
                  f"{b_count} normal ({b_count/total*100:.1f}%)")
    else:
        analyzer.run_analysis(args.metric)
    
    print(f"\nAnalysis complete! Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()