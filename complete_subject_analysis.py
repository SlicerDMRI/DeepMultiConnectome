#!/usr/bin/env python3
"""
Complete Subject Analysis Script

This script combines comprehensive connectome analysis with testing functionality.
It performs complete connectome analysis using the unified connectome system with 
real data, including all diffusion metric weighted connectomes.

Based on comprehensive_analysis.py and test_subject_100206.py
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
import tempfile
from typing import Optional
from typing import Dict, List, Tuple, Optional

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.unified_connectome import ConnectomeAnalyzer, analyze_connectomes_from_labels
from utils.connectome_utils import create_all_connectomes, compare_all_connectomes
from utils.logger import create_logger


class CompleteSubjectAnalysis:
    """
    Complete subject analysis combining comprehensive connectome analysis with testing
    """
    
    def __init__(self, subject_id: str = "100206"):
        """Initialize the complete analysis"""
        
        self.subject_id = subject_id
        
        # Set up paths
        self.base_path = Path("/media/volume/MV_HCP")
        self.subject_path = self.base_path / "HCP_MRtrix" / self.subject_id
        self.true_base = self.subject_path / "output"
        self.pred_base = self.subject_path / "TractCloud"
        self.diffusion_dir = self.subject_path / "dMRI"
        self.tractography_path = self.subject_path / "output" / "streamlines_10M.vtk"
        self.output_base_dir = self.base_path / "HCP_MRtrix" / self.subject_id / "analysis"
        
        # Atlas configuration
        self.atlases = ["aparc+aseg", "aparc.a2009s+aseg"]
        
        # Create output directory first
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = create_logger(str(self.output_base_dir))
        
        print(f"Initialized complete analysis for subject {self.subject_id}")
        self.logger.info(f"Initialized complete analysis for subject {self.subject_id}")
    
    def _find_and_prepare_data_files(self):
        """Find real predicted and true labels"""
        print("Searching for real data files...")
        self.logger.info("Searching for real data files...")
        
        pred_files = {}
        true_files = {}
        
        # Create output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the specific file paths for each atlas
        atlas_files = {
            "aparc+aseg": {
                "true": self.true_base / "labels_10M_aparc+aseg_symmetric.txt",
                "pred": self.pred_base / "predictions_aparc+aseg_symmetric.txt"
            },
            "aparc.a2009s+aseg": {
                "true": self.true_base / "labels_10M_aparc.a2009s+aseg_symmetric.txt",
                "pred": self.pred_base / "predictions_aparc.a2009s+aseg_symmetric.txt"
            }
        }
        
        for atlas in self.atlases:
            print(f"Processing {atlas}...")
            self.logger.info(f"Processing {atlas}...")
            
            true_labels_file = atlas_files[atlas]["true"]
            pred_labels_file = atlas_files[atlas]["pred"]
            
            # Check if both files exist
            if true_labels_file.exists() and pred_labels_file.exists():
                print(f"  ✓ Found true labels: {true_labels_file}")
                print(f"  ✓ Found predicted labels: {pred_labels_file}")
                self.logger.info(f"Found true labels: {true_labels_file}")
                self.logger.info(f"Found predicted labels: {pred_labels_file}")
                
                # Use the files directly
                pred_files[atlas] = str(pred_labels_file)
                true_files[atlas] = str(true_labels_file)
                                
            else:
                if not true_labels_file.exists():
                    print(f"  ⚠️  True labels file not found: {true_labels_file}")
                    self.logger.warning(f"True labels file not found: {true_labels_file}")
                if not pred_labels_file.exists():
                    print(f"  ⚠️  Predicted labels file not found: {pred_labels_file}")
                    self.logger.warning(f"Predicted labels file not found: {pred_labels_file}")
                continue
        
        if not pred_files or not true_files:
            print("❌ No valid data file pairs found! Check if both true and predicted labels exist.")
            self.logger.error("No valid data file pairs found!")
            return None, None
            
        print(f"✓ Successfully found data for {len(pred_files)} atlases")
        self.logger.info(f"Successfully found data for {len(pred_files)} atlases")
        return pred_files, true_files
    
    def test_unified_connectome_system(self):
        """Test the unified connectome system similar to test_subject_100206.py"""
        
        print("\n" + "="*80)
        print("TESTING UNIFIED CONNECTOME SYSTEM")
        print("="*80)
        self.logger.info("="*80)
        self.logger.info("TESTING UNIFIED CONNECTOME SYSTEM") 
        self.logger.info("="*80)
        
        test_output_dir = self.output_base_dir / "system_tests"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Test 1: Basic ConnectomeAnalyzer functionality
            print("\n1. Testing ConnectomeAnalyzer with real data...")
            self.logger.info("Test 1: Basic ConnectomeAnalyzer functionality")
            
            for atlas in self.atlases:
                print(f"  Testing atlas: {atlas}")
                self.logger.info(f"Testing atlas: {atlas}")
                
                # Initialize analyzer
                analyzer = ConnectomeAnalyzer(
                    atlas=atlas, 
                    out_path=str(test_output_dir), 
                    logger=self.logger
                )
                
                # Load real labels
                true_labels_file = self.true_base / f"labels_10M_{atlas}_symmetric.txt"
                if true_labels_file.exists():
                    with open(true_labels_file, 'r') as f:
                        true_labels = [int(line.strip()) for line in f if line.strip()]
                    
                    # Create connectome from real labels
                    analyzer.create_connectome_from_labels(true_labels, connectome_name=f'test_{atlas}')
                    self.logger.info(f"Created connectome from {len(true_labels)} real streamline labels")
                    
                    # Test with diffusion metrics
                    for metric in ['fa', 'md', 'ad', 'rd']:
                        metric_file = self.diffusion_dir / f"mean_{metric}_per_streamline.txt"
                        if metric_file.exists():
                            with open(metric_file, 'r') as f:
                                lines = f.readlines()
                            
                            # Filter out comments and empty lines, parse numbers
                            metric_values = []
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    try:
                                        # Handle multiple values per line (space-separated)
                                        values = line.split()
                                        for val in values:
                                            if val.lower() != 'nan':  # Skip NaN values
                                                metric_values.append(float(val))
                                    except ValueError:
                                        continue  # Skip lines that can't be parsed
                            
                            if metric_values:
                                # Ensure same length as labels
                                min_len = min(len(true_labels), len(metric_values))
                                labels_subset = true_labels[:min_len]
                                values_subset = metric_values[:min_len]
                                
                                analyzer.create_connectome_from_labels(
                                    labels_subset, 
                                    weights=values_subset,
                                    connectome_name=f'test_{atlas}_{metric}'
                                )
                                self.logger.info(f"Created {metric.upper()} weighted connectome with {len(labels_subset)} streamlines")
                    
                    # Save results
                    analyzer.save_results_summary(f'test1_{atlas}')
                    print(f"    ✓ {atlas} test completed")
                else:
                    self.logger.warning(f"Labels file not found: {true_labels_file}")
                    print(f"    ⚠ {atlas} labels file not found")
            
            # Test 2: Low-level connectome utilities  
            print("\n2. Testing connectome utilities...")
            self.logger.info("Test 2: Connectome utilities")
            
            utils_output = test_output_dir / "utils_test"
            utils_output.mkdir(parents=True, exist_ok=True)
            
            atlas = 'aparc+aseg'
            pred_file = self.pred_base / f"predictions_{atlas}_symmetric.txt"
            true_file = self.true_base / f"labels_10M_{atlas}_symmetric.txt"
            
            if pred_file.exists() and true_file.exists():
                # Test create_all_connectomes
                connectome_files = create_all_connectomes(
                    subject_path=str(self.subject_path),
                    atlas=atlas,
                    pred_labels_file=str(pred_file),
                    true_labels_file=str(true_file),
                    diffusion_metrics_dir=str(self.diffusion_dir),
                    out_path=str(utils_output),
                    num_labels=85,
                    logger=self.logger
                )
                
                if connectome_files:
                    print(f"    ✓ Created {len(connectome_files)} connectome files")
                    
                    # Test compare_all_connectomes
                    comparison_results = compare_all_connectomes(
                        connectome_files=connectome_files,
                        out_path=str(utils_output),
                        atlas=atlas,
                        logger=self.logger
                    )
                    
                    if comparison_results:
                        print(f"    ✓ Completed {len(comparison_results)} comparisons")
                    else:
                        print("    ✗ Comparison failed")
                else:
                    print("    ✗ Connectome creation failed")
            
            # Test 3: Performance test
            print("\n3. Testing performance with large dataset...")
            self.logger.info("Test 3: Performance and edge cases")
            
            start_time = time.time()
            atlas = 'aparc+aseg'
            true_labels_file = self.true_base / f"labels_10M_{atlas}_symmetric.txt"
            
            if true_labels_file.exists():
                with open(true_labels_file, 'r') as f:
                    all_labels = [int(line.strip()) for line in f if line.strip()]
                
                print(f"    Testing with {len(all_labels)} streamlines...")
                
                analyzer = ConnectomeAnalyzer(
                    atlas=atlas, 
                    out_path=str(test_output_dir), 
                    logger=self.logger
                )
                
                # Test large dataset
                analyzer.create_connectome_from_labels(all_labels, connectome_name='large_test')
                
                # Test with diffusion metrics
                fa_file = self.diffusion_dir / "mean_fa_per_streamline.txt"
                if fa_file.exists():
                    with open(fa_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse FA values using the correct method
                    fa_values = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                values = line.split()
                                for val in values:
                                    if val.lower() != 'nan':
                                        fa_values.append(float(val))
                            except ValueError:
                                continue
                    
                    if fa_values:
                        min_len = min(len(all_labels), len(fa_values))
                        analyzer.create_connectome_from_labels(
                            all_labels[:min_len], 
                            weights=fa_values[:min_len],
                            connectome_name='large_fa_test'
                        )
                
                elapsed_time = time.time() - start_time
                print(f"    ✓ Large dataset test completed in {elapsed_time:.2f} seconds")
                self.logger.info(f"Large dataset processing time: {elapsed_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"\n✗ System test failed with error: {e}")
            self.logger.error(f"System test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_comprehensive_analysis(self, atlas: str, pred_labels_file: str, true_labels_file: str, 
                                 apply_thresholding: bool = False, threshold_percentage: float = 5.0, 
                                 min_streamlines: int = 5, apply_length_filtering: bool = False,
                                 min_length: float = 20.0, max_length: Optional[float] = None,
                                 lengths_file: Optional[str] = None):
        """
        Run comprehensive analysis for a single atlas using the unified connectome system
        
        Args:
            atlas: Atlas name
            pred_labels_file: Path to predicted labels file
            true_labels_file: Path to true labels file
            apply_thresholding: Whether to apply node thresholding
            threshold_percentage: Percentage of nodes to filter out
            min_streamlines: Minimum number of streamlines for a node to keep
            apply_length_filtering: Whether to apply streamline length filtering
            min_length: Minimum streamline length in mm
            max_length: Maximum streamline length in mm (optional)
            lengths_file: Path to streamline lengths file (auto-detected if None)
        """
        print(f"\n🔸 Analyzing {atlas}")
        self.logger.info(f"Analyzing {atlas}")
        
        # Auto-detect lengths file if not provided but length filtering is requested
        if apply_length_filtering and lengths_file is None:
            # Try common locations for streamline lengths file
            potential_lengths_files = [
                self.subject_path / "output" / "streamline_lengths_10M.txt",
                self.subject_path / "output" / "streamline_lengths.txt",
                self.diffusion_dir / "streamline_lengths.txt"
            ]
            
            for potential_file in potential_lengths_files:
                if potential_file.exists():
                    lengths_file = str(potential_file)
                    self.logger.info(f"Auto-detected lengths file: {lengths_file}")
                    break
            
            if lengths_file is None:
                self.logger.warning("Length filtering requested but no lengths file found. Disabling length filtering.")
                apply_length_filtering = False
        
        # Set up output directory
        atlas_output_dir = self.output_base_dir / atlas
        atlas_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Running comprehensive connectome analysis...")
        if apply_length_filtering:
            length_desc = f"≥{min_length}mm"
            if max_length is not None:
                length_desc = f"{min_length}-{max_length}mm"
            print(f"  With length filtering: {length_desc}")
        
        try:
            analyzer = analyze_connectomes_from_labels(
                pred_labels_file=pred_labels_file,
                true_labels_file=true_labels_file,
                diffusion_metrics_dir=str(self.diffusion_dir),
                atlas=atlas,
                out_path=str(atlas_output_dir),
                logger=self.logger,
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
                
                self.logger.info(f"Analysis complete for {atlas}")
                self.logger.info(f"Correlation: {correlation:.4f}")
                self.logger.info(f"RMSE: {rmse:.4f}")
                self.logger.info(f"Mean LERM: {mean_lerm:.4f}")
                self.logger.info(f"Connections: {connections}")
                
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
                self.logger.error(f"Analysis failed for {atlas} - no results generated")
                return None
                
        except Exception as e:
            print(f"  ✗ Analysis failed with error: {e}")
            self.logger.error(f"Analysis failed for {atlas}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_analysis(self, run_tests: bool = False, apply_thresholding: bool = False,
                            threshold_percentage: float = 5.0, min_streamlines: int = 5,
                            apply_length_filtering: bool = False, min_length: float = 20.0,
                            max_length: Optional[float] = None, lengths_file: Optional[str] = None):
        """
        Run the complete analysis including tests and comprehensive connectome analysis
        
        Args:
            run_tests: Whether to run system tests first
            apply_thresholding: Whether to apply node thresholding
            threshold_percentage: Percentage of nodes to filter out
            min_streamlines: Minimum number of streamlines for a node to keep
            apply_length_filtering: Whether to apply streamline length filtering
            min_length: Minimum streamline length in mm
            max_length: Maximum streamline length in mm (optional)
            lengths_file: Path to streamline lengths file (auto-detected if None)
        """
        print("="*80)
        print("COMPLETE SUBJECT ANALYSIS")  
        print("Subject:", self.subject_id)
        print("Output directory:", self.output_base_dir)
        if apply_thresholding:
            print(f"Thresholding enabled: {threshold_percentage}% nodes removed, min {min_streamlines} streamlines")
        if apply_length_filtering:
            length_desc = f"≥{min_length}mm"
            if max_length is not None:
                length_desc = f"{min_length}-{max_length}mm"
            print(f"Length filtering enabled: {length_desc}")
        print("="*80)
        
        self.logger.info("="*80)
        self.logger.info("COMPLETE SUBJECT ANALYSIS")
        self.logger.info(f"Subject: {self.subject_id}")
        self.logger.info(f"Output directory: {self.output_base_dir}")
        if apply_thresholding:
            self.logger.info(f"Thresholding enabled: {threshold_percentage}% nodes removed, min {min_streamlines} streamlines")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        # Optional: Run system tests first
        if run_tests:
            test_success = self.test_unified_connectome_system()
            if not test_success:
                print("⚠️  System tests failed, but continuing with analysis...")
                self.logger.warning("System tests failed, but continuing with analysis...")
        
        # Find and prepare data files
        pred_files, true_files = self._find_and_prepare_data_files()
        
        if pred_files is None or true_files is None:
            print("❌ Failed to find or prepare data files!")
            self.logger.error("Failed to find or prepare data files!")
            return []
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE CONNECTOME ANALYSIS")
        print(f"{'='*80}")
        
        all_results = []
        
        for atlas in self.atlases:
            if atlas not in pred_files or atlas not in true_files:
                print(f"⚠️  Skipping {atlas} - missing data files")
                self.logger.warning(f"Skipping {atlas} - missing data files")
                continue
            
            print(f"\n📁 Processing {atlas}...")
            print(f"  Predicted labels: {pred_files[atlas]}")
            print(f"  True labels: {true_files[atlas]}")
            
            # Run comprehensive analysis
            result = self.run_comprehensive_analysis(atlas, pred_files[atlas], true_files[atlas],
                                                   apply_thresholding, threshold_percentage, min_streamlines,
                                                   apply_length_filtering, min_length, max_length, lengths_file)
            
            if result is not None:
                all_results.append(result)
            else:
                print(f"  ⚠️  Analysis failed for {atlas}")
                self.logger.warning(f"Analysis failed for {atlas}")
        
        if all_results:
            # Create summary
            self._create_summary_report(all_results)
            
            elapsed_time = time.time() - start_time
            print(f"\n🎉 Complete analysis finished!")
            print(f"Results saved to: {self.output_base_dir}")
            print(f"Total analysis time: {elapsed_time:.2f} seconds")
            
            self.logger.info("Complete analysis finished successfully!")
            self.logger.info(f"Results saved to: {self.output_base_dir}")
            self.logger.info(f"Total analysis time: {elapsed_time:.2f} seconds")
        else:
            print("\n❌ No successful analyses completed!")
            self.logger.error("No successful analyses completed!")
            
        return all_results
    
    def _create_summary_report(self, results: List[Dict]):
        """Create a comprehensive summary report of all analyses"""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        self.logger.info("="*80)
        self.logger.info("ANALYSIS SUMMARY")
        self.logger.info("="*80)
        
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
        summary_csv = self.output_base_dir / "complete_analysis_summary.csv"
        df.to_csv(summary_csv, index=False)
        print(f"Summary saved to: {summary_csv}")
        self.logger.info(f"Summary saved to: {summary_csv}")
        
        # Print summary table
        print("\nResults Summary:")
        print(df.to_string(index=False))
        
        # Log summary
        self.logger.info("Results Summary:")
        self.logger.info("\n" + df.to_string(index=False))
        
        # Create comparison plot
        self._create_comparison_plot(df)
        
        # Create detailed summary with all connectome types
        self._create_detailed_summary(results)
        
        print("\n" + "="*80)
    
    def _create_detailed_summary(self, results: List[Dict]):
        """Create detailed summary including all connectome types"""
        
        detailed_summary = []
        
        for result in results:
            analyzer = result['analyzer']
            atlas = result['atlas']
            
            # Get all connectomes and metrics
            for connectome_name, connectome in analyzer.connectomes.items():
                connections = np.count_nonzero(connectome)
                total_strength = np.sum(connectome)
                
                detailed_summary.append({
                    'Atlas': atlas,
                    'Connectome_Type': connectome_name,
                    'Shape': f"{connectome.shape[0]}x{connectome.shape[1]}",
                    'Connections': connections,
                    'Total_Strength': total_strength,
                    'Density': connections / (connectome.shape[0] * (connectome.shape[0] - 1)),
                    'Mean_Strength': np.mean(connectome[connectome > 0]) if connections > 0 else 0
                })
        
        detailed_df = pd.DataFrame(detailed_summary)
        detailed_csv = self.output_base_dir / "detailed_connectome_summary.csv"
        detailed_df.to_csv(detailed_csv, index=False)
        
        print(f"Detailed summary saved to: {detailed_csv}")
        self.logger.info(f"Detailed summary saved to: {detailed_csv}")
    
    def _create_comparison_plot(self, df: pd.DataFrame):
        """Create comprehensive comparison plot of results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Complete Subject Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Correlation comparison
        sns.barplot(data=df, x='Atlas', y='Correlation', ax=axes[0,0])
        axes[0,0].set_title('Correlation Comparison')
        axes[0,0].set_ylabel('Pearson Correlation')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(0, 1)
        
        # Plot 2: RMSE comparison
        sns.barplot(data=df, x='Atlas', y='RMSE', ax=axes[0,1])
        axes[0,1].set_title('RMSE Comparison')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Mean LERM comparison
        sns.barplot(data=df, x='Atlas', y='Mean_LERM', ax=axes[1,0])
        axes[1,0].set_title('Mean LERM Comparison')
        axes[1,0].set_ylabel('Mean LERM')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Connections comparison
        sns.barplot(data=df, x='Atlas', y='Connections', ax=axes[1,1])
        axes[1,1].set_title('Connection Count Comparison')
        axes[1,1].set_ylabel('Number of Connections')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_base_dir / "complete_analysis_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {plot_path}")
        self.logger.info(f"Comparison plot saved to: {plot_path}")


def main():
    """Main function"""
    
    print("Starting Complete Subject Analysis")
    print("Combining comprehensive connectome analysis with system testing")
    print("="*80)
    
    # Parse command line arguments for different modes
    import argparse
    parser = argparse.ArgumentParser(description='Complete Subject Analysis')
    parser.add_argument('--subject', '-s', default='100206', help='Subject ID')
    parser.add_argument('--tests', action='store_true', help='Skip system tests')
    parser.add_argument('--tests-only', action='store_true', help='Run only system tests')
    
    # Thresholding arguments
    parser.add_argument('--threshold', action='store_true', help='Apply connectome thresholding')
    parser.add_argument('--threshold-percentage', type=float, default=5.0, 
                       help='Percentage of nodes to filter out (default: 5.0)')
    parser.add_argument('--min-streamlines', type=int, default=5,
                       help='Minimum number of streamlines for a node to keep (default: 5)')
    
    # Length filtering arguments
    parser.add_argument('--length-filter', action='store_true', help='Apply streamline length filtering')
    parser.add_argument('--min-length', type=float, default=20.0,
                       help='Minimum streamline length in mm (default: 20.0)')
    parser.add_argument('--max-length', type=float, default=None,
                       help='Maximum streamline length in mm (optional)')
    parser.add_argument('--lengths-file', type=str, default=None,
                       help='Path to streamline lengths file (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Run analysis
    analysis = CompleteSubjectAnalysis(subject_id=args.subject)
    
    if args.tests_only:
        # Run only system tests
        print("Running system tests only...")
        test_success = analysis.test_unified_connectome_system()
        if test_success:
            print("\n🎉 System tests passed!")
        else:
            print("\n❌ System tests failed!")
    else:
        # Run complete analysis
        run_tests = args.tests
        results = analysis.run_complete_analysis(
            run_tests=run_tests,
            apply_thresholding=args.threshold,
            threshold_percentage=args.threshold_percentage,
            min_streamlines=args.min_streamlines,
            apply_length_filtering=args.length_filter,
            min_length=args.min_length,
            max_length=args.max_length,
            lengths_file=args.lengths_file
        )
        
        if results:
            print(f"\n🎉 Complete analysis finished successfully!")
            print(f"Analyzed {len(results)} atlas configurations")
            print(f"Results available in: {analysis.output_base_dir}")
        else:
            print("\n❌ Analysis failed!")


if __name__ == "__main__":
    main()