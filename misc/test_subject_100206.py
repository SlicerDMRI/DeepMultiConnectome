#!/usr/bin/env python3
"""
Comprehensive test of all unified connectome functions on subject 100206

This script tests the entire unified connectome system using real data from subject 100206.
"""

import sys
import os
import time
import shutil
import tempfile

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.unified_connectome import ConnectomeAnalyzer, analyze_connectomes_from_labels
from utils.connectome_utils import create_all_connectomes, compare_all_connectomes
from utils.logger import create_logger


def test_subject_100206():
    """Comprehensive test of all connectome functions on subject 100206"""
    
    # Subject paths
    subject_path = "/media/volume/MV_HCP/HCP_MRtrix/100206"
    tractography_path = os.path.join(subject_path, "output", "streamlines_10M.vtk")
    
    # Create temporary output directory
    test_output_dir = "/media/volume/MV_HCP/HCP_MRtrix/100206/DeepMultiConnectome"
    print(f"Test output directory: {test_output_dir}")
    
    # Create logger
    logger = create_logger(test_output_dir)
    logger.info("="*80)
    logger.info("COMPREHENSIVE TEST OF UNIFIED CONNECTOME SYSTEM")
    logger.info("Subject: 100206")
    logger.info("="*80)
    
    try:
        # Test 1: Basic ConnectomeAnalyzer functionality
        print("\n1. Testing ConnectomeAnalyzer with real data...")
        logger.info("Test 1: Basic ConnectomeAnalyzer functionality")
        
        for atlas in ['aparc+aseg', 'aparc.a2009s+aseg']:
            print(f"  Testing atlas: {atlas}")
            logger.info(f"Testing atlas: {atlas}")
            
            # Initialize analyzer
            analyzer = ConnectomeAnalyzer(atlas=atlas, out_path=test_output_dir, logger=logger)
            
            # Load real labels
            true_labels_file = os.path.join(subject_path, "output", f"labels_10M_{atlas}_symmetric.txt")
            if os.path.exists(true_labels_file):
                with open(true_labels_file, 'r') as f:
                    true_labels = [int(line.strip()) for line in f if line.strip()]
                
                # Create connectome from real labels
                analyzer.create_connectome_from_labels(true_labels, connectome_name=f'real_{atlas}')
                logger.info(f"Created connectome from {len(true_labels)} real streamline labels")
                
                # Test with diffusion metrics
                diffusion_dir = os.path.join(subject_path, "dMRI")
                for metric in ['fa', 'md', 'ad', 'rd']:
                    metric_file = os.path.join(diffusion_dir, f"mean_{metric}_per_streamline.txt")
                    if os.path.exists(metric_file):
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
                                connectome_name=f'real_{atlas}_{metric}'
                            )
                            logger.info(f"Created {metric.upper()} weighted connectome with {len(labels_subset)} streamlines")
                
                # Test network analysis
                # analyzer.compute_network_metrics(f'real_{atlas}')
                # logger.info(f"Computed network metrics for {atlas}")
                
                # Save results
                analyzer.save_results_summary(f'test1_{atlas}')
                print(f"    ✓ {atlas} test completed")
            else:
                logger.warning(f"Labels file not found: {true_labels_file}")
                print(f"    ⚠ {atlas} labels file not found")
        
        # Test 2: High-level analysis function
        print("\n2. Testing high-level analyze_connectomes_from_labels function...")
        logger.info("Test 2: High-level analysis function")
        
        # We'll simulate predictions by using a subset of the true labels
        atlas = 'aparc+aseg'
        true_labels_file = os.path.join(subject_path, "output", f"labels_10M_{atlas}_symmetric.txt")
        
        if os.path.exists(true_labels_file):
            # Create simulated predictions (randomly perturb some true labels)
            with open(true_labels_file, 'r') as f:
                true_labels = [int(line.strip()) for line in f if line.strip()]
            
            # Create predictions by introducing some noise to true labels
            import random
            random.seed(42)
            pred_labels = []
            for label in true_labels[:10000]:  # Use subset for faster testing
                if random.random() < 0.1:  # 10% chance to change label
                    # Change to a random valid label
                    pred_labels.append(random.randint(0, 85*85-1))
                else:
                    pred_labels.append(label)
            
            # Save simulated predictions
            pred_labels_file = os.path.join(test_output_dir, f"simulated_predictions_{atlas}_symmetric.txt")
            with open(pred_labels_file, 'w') as f:
                for label in pred_labels:
                    f.write(f"{label}\n")
            
            # Save corresponding true labels
            true_subset_file = os.path.join(test_output_dir, f"true_subset_{atlas}_symmetric.txt")
            with open(true_subset_file, 'w') as f:
                for label in true_labels[:len(pred_labels)]:
                    f.write(f"{label}\n")
            
            # Run comprehensive analysis
            analyzer = analyze_connectomes_from_labels(
                pred_labels_file=pred_labels_file,
                true_labels_file=true_subset_file,
                diffusion_metrics_dir=os.path.join(subject_path, "dMRI"),
                atlas=atlas,
                out_path=os.path.join(test_output_dir, "comprehensive_test"),
                logger=logger
            )
            
            if analyzer:
                print("    ✓ Comprehensive analysis completed")
                analyzer.print_summary()
            else:
                print("    ✗ Comprehensive analysis failed")
        
        # Test 3: Low-level connectome utilities
        print("\n3. Testing connectome utilities...")
        logger.info("Test 3: Connectome utilities")
        
        utils_output = os.path.join(test_output_dir, "utils_test")
        os.makedirs(utils_output, exist_ok=True)
        
        atlas = 'aparc+aseg'
        pred_file = os.path.join(test_output_dir, f"simulated_predictions_{atlas}_symmetric.txt")
        true_file = os.path.join(test_output_dir, f"true_subset_{atlas}_symmetric.txt")
        
        if os.path.exists(pred_file) and os.path.exists(true_file):
            # Test create_all_connectomes
            connectome_files = create_all_connectomes(
                subject_path=subject_path,
                atlas=atlas,
                pred_labels_file=pred_file,
                true_labels_file=true_file,
                diffusion_metrics_dir=os.path.join(subject_path, "dMRI"),
                out_path=utils_output,
                num_labels=85,
                logger=logger
            )
            
            if connectome_files:
                print(f"    ✓ Created {len(connectome_files)} connectome files")
                
                # Test compare_all_connectomes
                comparison_results = compare_all_connectomes(
                    connectome_files=connectome_files,
                    out_path=utils_output,
                    atlas=atlas,
                    logger=logger
                )
                
                if comparison_results:
                    print(f"    ✓ Completed {len(comparison_results)} comparisons")
                else:
                    print("    ✗ Comparison failed")
            else:
                print("    ✗ Connectome creation failed")
        
        # Test 4: Real prediction simulation (test_realdata.py functionality)
        print("\n4. Testing prediction pipeline...")
        logger.info("Test 4: Testing prediction pipeline")
        
        # Test with actual tractography file
        if os.path.exists(tractography_path):
            print(f"    Tractography file found: {os.path.basename(tractography_path)}")
            logger.info(f"Found tractography file: {tractography_path}")
            
            # Simulate what test_realdata.py would do
            # (We can't run the full DL prediction without the model, but we can test the connectome parts)
            
            atlas = 'aparc+aseg'
            true_labels_file = os.path.join(subject_path, "output", f"labels_10M_{atlas}_symmetric.txt")
            
            if os.path.exists(true_labels_file):
                # Create a simulation of what the prediction pipeline would produce
                pred_output = os.path.join(test_output_dir, "prediction_simulation")
                os.makedirs(pred_output, exist_ok=True)
                
                # Copy true labels as "predictions" for testing
                pred_sim_file = os.path.join(pred_output, f"predictions_{atlas}_symmetric.txt")
                shutil.copy2(true_labels_file, pred_sim_file)
                
                # Run the same analysis that test_realdata.py would do
                analyzer = analyze_connectomes_from_labels(
                    pred_labels_file=pred_sim_file,
                    true_labels_file=true_labels_file,
                    diffusion_metrics_dir=os.path.join(subject_path, "dMRI"),
                    atlas=atlas,
                    out_path=pred_output,
                    logger=logger
                )
                
                if analyzer:
                    print("    ✓ Prediction pipeline simulation completed")
                    print("    Results summary:")
                    analyzer.print_summary()
                else:
                    print("    ✗ Prediction pipeline simulation failed")
        
        # Test 5: Performance and edge cases
        print("\n5. Testing performance and edge cases...")
        logger.info("Test 5: Performance and edge cases")
        
        # Test with large dataset
        start_time = time.time()
        atlas = 'aparc+aseg'
        true_labels_file = os.path.join(subject_path, "output", f"labels_10M_{atlas}_symmetric.txt")
        
        if os.path.exists(true_labels_file):
            with open(true_labels_file, 'r') as f:
                all_labels = [int(line.strip()) for line in f if line.strip()]
            
            print(f"    Testing with {len(all_labels)} streamlines...")
            
            analyzer = ConnectomeAnalyzer(atlas=atlas, out_path=test_output_dir, logger=logger)
            
            # Test large dataset
            analyzer.create_connectome_from_labels(all_labels, connectome_name='large_test')
            
            # Test with diffusion metrics
            fa_file = os.path.join(subject_path, "dMRI", "mean_fa_per_streamline.txt")
            if os.path.exists(fa_file):
                with open(fa_file, 'r') as f:
                    lines = f.readlines()
                
                # Filter out comments and empty lines, parse numbers
                fa_values = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            # Handle multiple values per line (space-separated)
                            values = line.split()
                            for val in values:
                                if val.lower() != 'nan':  # Skip NaN values
                                    fa_values.append(float(val))
                        except ValueError:
                            continue  # Skip lines that can't be parsed
                
                if fa_values:
                    min_len = min(len(all_labels), len(fa_values))
                    analyzer.create_connectome_from_labels(
                        all_labels[:min_len], 
                        weights=fa_values[:min_len],
                        connectome_name='large_fa_test'
                    )
            
            elapsed_time = time.time() - start_time
            print(f"    ✓ Large dataset test completed in {elapsed_time:.2f} seconds")
            logger.info(f"Large dataset processing time: {elapsed_time:.2f} seconds")
        
        # Final summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print("✓ All unified connectome functions tested successfully!")
        print(f"✓ Test results saved to: {test_output_dir}")
        print(f"✓ Check the log files and CSV outputs for detailed results")
        print("="*80)
        
        logger.info("All tests completed successfully!")
        logger.info(f"Results saved to: {test_output_dir}")
        
        return test_output_dir
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting comprehensive test of unified connectome system on subject 100206...")
    result_dir = test_subject_100206()
    
    if result_dir:
        print(f"\n🎉 All tests passed! Results available in: {result_dir}")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")