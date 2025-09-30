#!/usr/bin/env python3
"""
Example script showing how to use the connectome utilities

This demonstrates the basic usage of the connectome creation and comparison tools.
"""

import sys
import os
sys.path.append('../')

from utils.connectome_utils import create_all_connectomes, compare_all_connectomes, ConnectomeBuilder, ConnectomeComparator
from utils.logger import create_logger
from utils.connectome_config import ATLAS_CONFIG


def example_usage():
    """
    Example showing how to create and compare connectomes
    """
    
    # Example paths (adjust these to your actual data)
    subject_path = "/media/volume/MV_HCP/HCP_MRtrix/100206"
    atlas = "aparc+aseg"
    out_path = "/media/volume/HCP_diffusion_MV/DeepMultiConnectome/example_output"
    
    # Create output directory
    os.makedirs(out_path, exist_ok=True)
    
    # Create logger
    logger = create_logger(out_path)
    logger.info("Starting connectome analysis example")
    
    # Get number of labels for this atlas
    num_labels_atlas = ATLAS_CONFIG[atlas]['num_labels']
    
    # File paths
    pred_labels_file = os.path.join(out_path, f'predictions_{atlas}_symmetric.txt')
    true_labels_file = os.path.join(subject_path, 'output', f'labels_10M_{atlas}_symmetric.txt')
    diffusion_metrics_dir = os.path.join(subject_path, 'dMRI')
    
    # Check if files exist
    if not os.path.exists(pred_labels_file):
        logger.error(f"Predicted labels file not found: {pred_labels_file}")
        logger.info("Please run the test_realdata.py script first to generate predictions")
        return
    
    if not os.path.exists(true_labels_file):
        logger.error(f"True labels file not found: {true_labels_file}")
        logger.info("Please make sure the tractography processing has been completed")
        return
    
    try:
        # Method 1: Use the high-level functions (recommended)
        logger.info("=== Method 1: Using high-level functions ===")
        
        # Create all connectomes
        connectome_files = create_all_connectomes(
            subject_path=subject_path,
            atlas=atlas,
            pred_labels_file=pred_labels_file,
            true_labels_file=true_labels_file,
            diffusion_metrics_dir=diffusion_metrics_dir,
            out_path=out_path,
            num_labels=num_labels_atlas,
            logger=logger
        )
        
        if connectome_files:
            # Compare all connectomes
            comparison_results = compare_all_connectomes(
                connectome_files=connectome_files,
                out_path=out_path,
                atlas=atlas,
                logger=logger
            )
            
            # Print results
            print("\nComparison Results Summary:")
            print("=" * 50)
            for comparison_name, results in comparison_results.items():
                print(f"{comparison_name}:")
                print(f"  Pearson r: {results['pearson_r']:.4f}")
                print(f"  Spearman r: {results['spearman_r']:.4f}")
                print(f"  Mean LERM: {results['mean_lerm']:.4f}")
                print(f"  RMSE: {results['RMSE']:.4f}")
                print("-" * 30)
        
        # Method 2: Use the low-level classes for more control
        logger.info("\n=== Method 2: Using low-level classes ===")
        
        # Create connectome builder
        builder = ConnectomeBuilder(num_labels_atlas, out_path, logger)
        
        # Load labels
        pred_labels = builder.load_streamline_labels(pred_labels_file)
        true_labels = builder.load_streamline_labels(true_labels_file)
        
        if pred_labels is not None and true_labels is not None:
            # Create a simple NoS connectome
            pred_nos = builder.build_connectome_matrix(pred_labels)
            true_nos = builder.build_connectome_matrix(true_labels)
            
            # Save connectomes
            builder.save_connectome(pred_nos, f'example_pred_nos_{atlas}.csv')
            builder.save_connectome(true_nos, f'example_true_nos_{atlas}.csv')
            
            # Create comparator
            comparator = ConnectomeComparator(out_path, logger)
            
            # Compare connectomes
            results = comparator.compare_connectomes(pred_nos, true_nos, f'example_nos_{atlas}')
            
            # Create plot
            fig = comparator.plot_comparison(pred_nos, true_nos, f'example_nos_{atlas}')
            
            print(f"\nExample NoS connectome comparison:")
            print(f"Pearson r: {results['pearson_r']:.4f}")
            print(f"Spearman r: {results['spearman_r']:.4f}")
            print(f"Mean LERM: {results['mean_lerm']:.4f}")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_connectome_types():
    """
    Demonstrate different types of connectomes that can be created
    """
    from utils.connectome_config import CONNECTOME_TYPES, DIFFUSION_METRICS
    
    print("Available connectome types:")
    print("=" * 40)
    
    for connectome_type, config in CONNECTOME_TYPES.items():
        print(f"{connectome_type.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Units: {config['units']}")
        if config['weight_file']:
            print(f"  Weight file: {config['weight_file']}")
        else:
            print(f"  Weight file: None (count-based)")
        print()
    
    print("Available diffusion metrics:")
    print("=" * 40)
    
    for metric, config in DIFFUSION_METRICS.items():
        print(f"{metric.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Units: {config['units']}")
        print(f"  Expected range: {config['range']}")
        print(f"  Filename: {config['filename']}")
        print()


if __name__ == "__main__":
    print("Connectome Utilities Example")
    print("=" * 50)
    
    # Show available connectome types
    demonstrate_connectome_types()
    
    # Run the example
    example_usage()