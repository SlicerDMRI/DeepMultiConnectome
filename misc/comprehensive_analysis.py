#!/usr/bin/env python3
"""
Comprehensive Connectome Analysis Script

This script performs a complete connectome analysis using the unified connectome system.
It creates connectomes, computes metrics, generates visualizations, and compares predicted vs true connectomes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import shutil
import random
from typing import Dict, List, Tuple

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.unified_connectome import ConnectomeAnalyzer


class ComprehensiveConnectomeAnalysis:
    """
    Comprehensive connectome analysis
    """
    
    def __init__(self, subject_id: str = "100206"):
        """Initialize the analysis"""
        
        self.subject_id = subject_id
        
        # Set up paths - using the correct paths for real data
        self.base_path = Path("/media/volume/MV_HCP")
        self.subject_path = self.base_path / "HCP_MRtrix" / self.subject_id
        self.true_base = self.subject_path / "output"
        self.pred_base = self.subject_path / "TractCloud"
        self.diffusion_dir = self.subject_path / "dMRI"
        self.output_base_dir = self.base_path / "HCP_MRtrix" / self.subject_id / "analysis"
        
        # Atlas configuration
        self.atlases = ["aparc+aseg", "aparc.a2009s+aseg"]
        
        print(f"Initialized comprehensive analysis for subject {self.subject_id}")
    
    def _find_and_prepare_data_files(self):
        """Find real predicted and true labels"""
        print("Searching for real data files...")
        
        pred_files = {}
        true_files = {}
        
        # Create output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the specific file paths for each atlas
        atlas_files = {
            "aparc+aseg": {
                "true": self.true_base / "labels_10M_aparc+aseg_symmetric.txt",  # Assuming this exists based on pattern
                "pred": self.pred_base / "predictions_aparc+aseg_symmetric.txt"
            },
            "aparc.a2009s+aseg": {
                "true": self.true_base / "labels_10M_aparc.a2009s+aseg_symmetric.txt",
                "pred": self.pred_base / "predictions_aparc.a2009s+aseg_symmetric.txt"
            }
        }
        
        for atlas in self.atlases:
            print(f"Processing {atlas}...")
            
            true_labels_file = atlas_files[atlas]["true"]
            pred_labels_file = atlas_files[atlas]["pred"]
            
            # Check if both files exist
            if true_labels_file.exists() and pred_labels_file.exists():
                print(f"  ✓ Found true labels: {true_labels_file}")
                print(f"  ✓ Found predicted labels: {pred_labels_file}")
                
                # Use the files directly
                pred_files[atlas] = str(pred_labels_file)
                true_files[atlas] = str(true_labels_file)
                                
            else:
                if not true_labels_file.exists():
                    print(f"  ⚠️  True labels file not found: {true_labels_file}")
                if not pred_labels_file.exists():
                    print(f"  ⚠️  Predicted labels file not found: {pred_labels_file}")
                continue
        
        if not pred_files or not true_files:
            print("❌ No valid data file pairs found! Check if both true and predicted labels exist.")
            return None, None
            
        print(f"✓ Successfully found data for {len(pred_files)} atlases")
        return pred_files, true_files
    
    def run_single_analysis(self, atlas: str, pred_labels_file: str, true_labels_file: str):
        """
        Run analysis for a single atlas using the unified connectome system
        
        Args:
            atlas: Atlas name
            pred_labels_file: Path to predicted labels file
            true_labels_file: Path to true labels file
        """
        print(f"\n🔸 Analyzing {atlas}")
        
        # Set up output directory
        atlas_output_dir = self.output_base_dir / atlas
        atlas_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the high-level analysis function from unified_connectome
        from utils.unified_connectome import analyze_connectomes_from_labels
        
        print(f"  Running comprehensive connectome analysis...")
        
        try:
            analyzer = analyze_connectomes_from_labels(
                pred_labels_file=pred_labels_file,
                true_labels_file=true_labels_file,
                diffusion_metrics_dir=str(self.diffusion_dir),
                atlas=atlas,
                out_path=str(atlas_output_dir),
                logger=None,  # Could add logger if needed
                compute_network_advanced=False,  # Keep it fast
                compute_network_centrality=False,
                compute_network_community=False
            )
            
            # Extract key results from the analyzer
            if analyzer and analyzer.metrics:
                # Get the main comparison results
                comparison_key = 'nos_comparison'  # Number of streamlines comparison
                if comparison_key in analyzer.metrics:
                    metrics = analyzer.metrics[comparison_key]
                    correlation = metrics.get('pearson_r', np.nan)
                    rmse = metrics.get('rmse', np.nan)
                    mean_lerm = metrics.get('mean_lerm', np.nan)
                else:
                    # Fallback: get first available comparison
                    first_comparison = list(analyzer.metrics.keys())[0]
                    metrics = analyzer.metrics[first_comparison]
                    correlation = metrics.get('pearson_r', np.nan)
                    rmse = metrics.get('rmse', np.nan)
                    mean_lerm = metrics.get('mean_lerm', np.nan)
                
                # Get connection count from true connectome
                if 'true_nos' in analyzer.connectomes:
                    connections = np.count_nonzero(analyzer.connectomes['true_nos'])
                else:
                    connections = np.nan
                
                print(f"  ✓ Analysis complete!")
                print(f"    Correlation: {correlation:.4f}")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    Mean LERM: {mean_lerm:.4f}")
                print(f"    Connections: {connections}")
                
                return {
                    'atlas': atlas,
                    'correlation': correlation,
                    'rmse': rmse,
                    'mean_lerm': mean_lerm,
                    'connections': connections,
                    'analyzer': analyzer
                }
            else:
                print(f"  ✗ Analysis failed - no results generated")
                return None
                
        except Exception as e:
            print(f"  ✗ Analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_analysis(self):
        """
        Run the complete analysis using real data files
        """
        print("="*80)
        print("COMPREHENSIVE CONNECTOME ANALYSIS")  
        print("Using real predicted and true labels")
        print("Subject:", self.subject_id)
        print("Output directory:", self.output_base_dir)
        print("="*80)
        
        # Find and prepare data files
        pred_files, true_files = self._find_and_prepare_data_files()
        
        if pred_files is None or true_files is None:
            print("❌ Failed to find or prepare data files!")
            return []
        
        all_results = []
        
        for atlas in self.atlases:
            if atlas not in pred_files or atlas not in true_files:
                print(f"⚠️  Skipping {atlas} - missing data files")
                continue
            
            print(f"\n📁 Processing {atlas}...")
            print(f"  Predicted labels: {pred_files[atlas]}")
            print(f"  True labels: {true_files[atlas]}")
            
            # Run analysis using file paths (not loading into memory)
            result = self.run_single_analysis(atlas, pred_files[atlas], true_files[atlas])
            
            if result is not None:
                all_results.append(result)
            else:
                print(f"  ⚠️  Analysis failed for {atlas}")
        
        if all_results:
            # Create summary
            self._create_summary_report(all_results)
            
            print("\n🎉 Complete analysis finished!")
            print(f"Results saved to: {self.output_base_dir}")
        else:
            print("\n❌ No successful analyses completed!")
            
        return all_results
    
    def _create_summary_report(self, results: List[Dict]):
        """Create a summary report of all analyses"""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append({
                'Atlas': result['atlas'],
                'Correlation': result['correlation'],
                'RMSE': result['rmse'],
                'Mean_LERM': result.get('mean_lerm', np.nan),
                'Connections': result['connections']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save summary CSV
        summary_csv = self.output_base_dir / "analysis_summary.csv"
        df.to_csv(summary_csv, index=False)
        print(f"Summary saved to: {summary_csv}")
        
        # Print summary table
        print("\nResults Summary:")
        print(df.to_string(index=False))
        
        # Create comparison plot
        self._create_comparison_plot(df)
        
        print("\n" + "="*80)
    
    def _create_comparison_plot(self, df: pd.DataFrame):
        """Create comparison plot of results"""
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        fig.suptitle('Analysis Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Correlation comparison
        sns.barplot(data=df, x='Atlas', y='Correlation', ax=axes[0])
        axes[0].set_title('Correlation Comparison')
        axes[0].set_ylabel('Pearson Correlation')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: RMSE comparison
        sns.barplot(data=df, x='Atlas', y='RMSE', ax=axes[1])
        axes[1].set_title('RMSE Comparison')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Mean LERM comparison
        sns.barplot(data=df, x='Atlas', y='Mean_LERM', ax=axes[2])
        axes[2].set_title('Mean LERM Comparison')
        axes[2].set_ylabel('Mean LERM')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Connections comparison
        sns.barplot(data=df, x='Atlas', y='Connections', ax=axes[3])
        axes[3].set_title('Connection Count Comparison')
        axes[3].set_ylabel('Number of Connections')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_base_dir / "analysis_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {plot_path}")


def main():
    """Main function"""
    
    print("Starting Comprehensive Connectome Analysis")
    print("Using real predicted and true streamline labels")
    print("="*80)
    
    # Run analysis
    analysis = ComprehensiveConnectomeAnalysis(subject_id="100206")
    start_time = time.time()
    
    results = analysis.run_complete_analysis()
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed_time:.2f} seconds")
    print(f"Analyzed {len(results)} configurations")


if __name__ == "__main__":
    main()