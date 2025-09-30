#!/usr/bin/env python3
"""
Standalone script for creating and comparing connectomes from tractography predictions

This script can be used independently to:
1. Create connectomes from predicted and ground truth labels
2. Weight connectomes with diffusion metrics (FA, MD, AD, RD)
3. Compare predicted vs ground truth connectomes
4. Generate comparison plots and statistics

Usage:
python create_connectomes.py --subject_path /path/to/subject --atlas aparc+aseg --out_path /path/to/output
"""

import argparse
import os
import sys
sys.path.append('../')

from utils.connectome_utils import create_all_connectomes, compare_all_connectomes
from utils.logger import create_logger


def main():
    parser = argparse.ArgumentParser(description="Create and compare connectomes from tractography predictions")
    
    # Required arguments
    parser.add_argument('--subject_path', type=str, required=True,
                        help='Path to subject directory containing tractography and diffusion metrics')
    parser.add_argument('--atlas', type=str, required=True, choices=['aparc+aseg', 'aparc.a2009s+aseg'],
                        help='Atlas used for parcellation')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Output directory for connectomes and analysis')
    
    # Optional arguments
    parser.add_argument('--pred_labels_file', type=str, default=None,
                        help='Path to predicted labels file (if not in standard location)')
    parser.add_argument('--true_labels_file', type=str, default=None,
                        help='Path to true labels file (if not in standard location)')
    parser.add_argument('--diffusion_metrics_dir', type=str, default=None,
                        help='Directory containing diffusion metric files (default: same as subject_path/dMRI)')
    parser.add_argument('--skip_comparison', action='store_true',
                        help='Skip connectome comparison (only create connectomes)')
    parser.add_argument('--create_plots', action='store_true', default=True,
                        help='Create comparison plots')
    
    args = parser.parse_args()
    
    # Set up paths
    if not os.path.exists(args.subject_path):
        print(f"Error: Subject path does not exist: {args.subject_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.out_path, exist_ok=True)
    
    # Create logger
    logger = create_logger(args.out_path)
    logger.info("Starting connectome creation and analysis")
    logger.info(f"Subject path: {args.subject_path}")
    logger.info(f"Atlas: {args.atlas}")
    logger.info(f"Output path: {args.out_path}")
    
    # Set default file paths if not provided
    if args.pred_labels_file is None:
        args.pred_labels_file = os.path.join(args.out_path, f'predictions_{args.atlas}_symmetric.txt')
    
    if args.true_labels_file is None:
        args.true_labels_file = os.path.join(args.subject_path, 'output', f'labels_10M_{args.atlas}_symmetric.txt')
        # Alternative location
        if not os.path.exists(args.true_labels_file):
            args.true_labels_file = os.path.join(args.subject_path, 'dMRI', f'labels_10M_{args.atlas}_symmetric.txt')
    
    if args.diffusion_metrics_dir is None:
        args.diffusion_metrics_dir = os.path.join(args.subject_path, 'dMRI')
    
    # Check if required files exist
    if not os.path.exists(args.pred_labels_file):
        logger.error(f"Predicted labels file not found: {args.pred_labels_file}")
        return 1
    
    if not os.path.exists(args.true_labels_file):
        logger.error(f"True labels file not found: {args.true_labels_file}")
        return 1
    
    if not os.path.exists(args.diffusion_metrics_dir):
        logger.error(f"Diffusion metrics directory not found: {args.diffusion_metrics_dir}")
        return 1
    
    # Set number of labels based on atlas
    num_labels = {"aparc+aseg": 85, "aparc.a2009s+aseg": 165}
    num_labels_atlas = num_labels[args.atlas]
    
    try:
        # Create all connectomes
        logger.info("Creating connectomes...")
        connectome_files = create_all_connectomes(
            subject_path=args.subject_path,
            atlas=args.atlas,
            pred_labels_file=args.pred_labels_file,
            true_labels_file=args.true_labels_file,
            diffusion_metrics_dir=args.diffusion_metrics_dir,
            out_path=args.out_path,
            num_labels=num_labels_atlas,
            logger=logger
        )
        
        if connectome_files:
            logger.info(f"Successfully created {len(connectome_files)} connectome files")
            
            # List created files
            for key, filepath in connectome_files.items():
                if filepath:
                    logger.info(f"  {key}: {os.path.basename(filepath)}")
            
            # Compare connectomes if not skipped
            if not args.skip_comparison:
                logger.info("Comparing predicted vs ground truth connectomes...")
                comparison_results = compare_all_connectomes(
                    connectome_files=connectome_files,
                    out_path=args.out_path,
                    atlas=args.atlas,
                    logger=logger
                )
                
                # Log summary of results
                logger.info("Connectome comparison summary:")
                logger.info("-" * 50)
                for comparison_name, results in comparison_results.items():
                    logger.info(f"{comparison_name}:")
                    logger.info(f"  Pearson correlation: {results['pearson_r']:.4f} (p={results['pearson_p']:.2e})")
                    logger.info(f"  Spearman correlation: {results['spearman_r']:.4f} (p={results['spearman_p']:.2e})")
                    logger.info(f"  Mean LERM: {results['mean_lerm']:.4f}")
                    logger.info(f"  RMSE: {results['RMSE']:.4f}")
                    logger.info(f"  MAE: {results['MAE']:.4f}")
                    logger.info("-" * 30)
                
                logger.info(f"Comparison summary saved to: connectome_comparison_summary_{args.atlas}.csv")
                if args.create_plots:
                    logger.info("Comparison plots saved to output directory")
            
            logger.info("Connectome analysis completed successfully!")
            return 0
            
        else:
            logger.error("Failed to create connectomes")
            return 1
            
    except Exception as e:
        logger.error(f"Error during connectome analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)