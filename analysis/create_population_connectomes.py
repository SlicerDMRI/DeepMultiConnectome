#!/usr/bin/env python3
"""
Create Population Population Connectomes

This script computes population-weighted average connectomes from the training set.
It handles multiple connectome types (NOS, FA, SIFT2, etc.) and atlases (aparc+aseg, etc.).

Usage:
    python3 create_population_connectomes.py [--base-path PATH] [--force-recompute]
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

# Add path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from utils.unified_connectome import ConnectomeAnalyzer
from utils.connectome_utils import ConnectomeBuilder
from utils.logger import create_logger

class PopulationConnectomeCreator:
    """
    Creates population average connectomes from training set subjects.
    """
    
    def __init__(self, base_path: str = "/media/volume/MV_HCP", out_path: str = '/media/volume/HCP_diffusion_MV/DeepMultiConnectome/analysis'):
        """Initialize the population connectome creator"""
        
        self.base_path = Path(base_path)

        # Subject lists - only need training subjects
        self.train_subjects_file = self.base_path / "subjects_tractography_output_1000_train_200.txt"
        
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
        
        print(f"Initialized population connectome creator")
        print(f"Training subjects: {len(self.train_subjects)}")
        print(f"Output directory: {self.output_dir}")
        
        self.logger.info(f"Initialized population connectome creator")
        self.logger.info(f"Training subjects: {len(self.train_subjects)}")
    
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
            # pred_base and diffusion_dir removed as not needed for training set averaging
        }
    
    def _load_subject_connectome(self, subject_id: str, atlas: str,
                                metric: str = "nos", connectome_type: str = "true") -> Optional[np.ndarray]:
        """
        Load a connectome for a specific subject and atlas
        
        Args:
            subject_id: Subject identifier
            atlas: Atlas name
            metric: Connectome metric ('nos', 'fa', 'sift2')
            connectome_type: 'true' for ground truth (only relevant one here)
        
        Returns:
            Connectome matrix or None if loading failed
        """
        paths = self._get_subject_paths(subject_id)
        
        try:
            # Determine filename based on metric for true connectomes
            if metric == 'nos':
                filename = f"connectome_matrix_{atlas}.csv"
            elif metric == 'sift2':
                filename = f"connectome_matrix_SIFT_sum_{atlas}.csv"
            elif metric in ['fa', 'md', 'ad', 'rd']:
                filename = f"connectome_matrix_{metric.upper()}_mean_{atlas}.csv"
            else:
                filename = f"connectome_matrix_{atlas}_{metric}.csv"

            # Try to load pre-computed connectome matrix first
            connectome_file = paths['true_base'] / filename
            
            if connectome_file.exists():
                # Load pre-computed connectome matrix
                # Read without header/index to preserve dimensions of raw matrix files
                connectome_df = pd.read_csv(connectome_file, header=None)
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
            
            self.logger.warning(f"No connectome data found for {subject_id} {atlas} ({connectome_type})")
            return None
                
        except Exception as e:
            self.logger.error(f"Error loading connectome for {subject_id} {atlas}: {e}")
            return None
    
    def compute_population_averages(self, force_recompute: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
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
                    
                    connectome = self._load_subject_connectome(subject_id, atlas, metric=connectome_type)
                    
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

                # Save average connectome as CSV (raw matrix, no headers/indices)
                pd.DataFrame(average_connectome).to_csv(cache_file_csv, header=False, index=False)
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

    def run(self, force_recompute: bool = False):
        """Run the population connectome creation pipeline"""
        try:
            average_connectomes = self.compute_population_averages(force_recompute=force_recompute)
            
            if average_connectomes:
                self._create_connectome_visualizations(average_connectomes)
                print("\nPopulation connectome creation completed successfully!")
            else:
                print("\nNo connectomes were computed.")
                
        except Exception as e:
            print(f"Error running pipeline: {e}")
            self.logger.error(f"Error running pipeline: {e}")
            traceback.print_exc()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create Population Average Connectomes')
    parser.add_argument('--base-path', default='/media/volume/MV_HCP', 
                       help='Base path for data (default: /media/volume/MV_HCP)')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation of cached results')
    
    args = parser.parse_args()
    
    # Initialize creator
    creator = PopulationConnectomeCreator(base_path=args.base_path)
    
    # Run
    creator.run(force_recompute=args.force_recompute)

if __name__ == "__main__":
    main()
