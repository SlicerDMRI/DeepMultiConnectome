#!/usr/bin/env python3
"""
Filter Streamlines by Minimum Length and Rebuild Connectomes

This script:
1. Filters streamlines based on minimum length from `streamline_lengths_10M.txt`
2. Updates `labels_10M_aparc+aseg.txt` (and a2009s) accordingly
3. Reconstructs connectomes:
   - NOS (count)
   - FA, SIFT2 (weighted, if available)

Usage:
    python3 filter_streamlines_minlength.py --subject-id <ID> --min-length <mm> --atlas <atlas_name>
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
sys.path.append(str(Path("/media/volume/HCP_diffusion_MV/DeepMultiConnectome")))

from utils.connectome_utils import ConnectomeBuilder
from utils.logger import create_logger

def load_labels_file(labels_file):
    """
    Load labels files efficiently using pandas.
    Handles comments starting with #.
    """
    try:
        # Fast load, assume all numeric
        # sep=r'\s+' handles space or tab
        # comment='#' skips lines starting with #
        df = pd.read_csv(labels_file, sep=r'\s+', comment='#', header=None, dtype=np.int32)
        return df.values
    except Exception:
        # Fallback if there's a header or mixed types, though likely unsupported
        df = pd.read_csv(labels_file, sep=r'\s+', comment='#', header=None)
        return df.values.astype(np.int32)

def fast_load_txt(filepath):
    """
    Fast load single column (or space-separated) numeric file.
    Handles comment lines starting with # (e.g. MRtrix command history).
    """
    try:
        with open(filepath, 'r') as f:
            # Read lines, filtering out comments
            # This is safer than f.read() when headers exist
            content = " ".join([line for line in f if not line.lstrip().startswith('#')])
            return np.fromstring(content, sep=' ')
    except Exception as e:
        print(f"Error loading {filepath}: {e}, using pandas fallback.")
        # Fallback (slow), ensure comment handling
        return pd.read_csv(filepath, header=None, sep=r'\s+', comment='#').values.flatten()

def filter_and_rebuild_connectomes(subject_id, min_length_mm, atlas_name, base_path="/media/volume/MV_HCP"):
    
    base_path = Path(base_path)
    # Correct paths based on context
    # /media/volume/MV_HCP/HCP_MRtrix/100206/output/streamline_lengths_10M.txt
    # /media/volume/MV_HCP/HCP_MRtrix/100206/output/labels_10M_aparc+aseg.txt
    
    subject_dir = base_path / "HCP_MRtrix" / subject_id
    output_dir = subject_dir / "output"
    
    # 1. Define input files
    lengths_file = output_dir / "streamline_lengths_10M.txt"
    labels_file = output_dir / f"labels_10M_{atlas_name}.txt"
    
    # Define predictions file
    # Format: /media/volume/MV_HCP/HCP_MRtrix/100206/TractCloud/predictions_aparc+aseg.txt
    pred_file = subject_dir / "TractCloud" / f"predictions_{atlas_name}.txt"
    
    # Also need diffusion metrics for weighted connectomes (if they exist)
    # Typically in dMRI folder or output folder
    # Based on streamline_length_thresholding.py context:
    # self.fa_file = self.subject_path / "dMRI" / "mean_fa_per_streamline.txt"
    dmri_dir = subject_dir / "dMRI"
    fa_file = dmri_dir / "mean_fa_per_streamline.txt"
    sift_file = dmri_dir / "sift2_weights.txt" 
    
    print(f"Processing Subject: {subject_id}")
    print(f"Atlas: {atlas_name}")
    print(f"Min Length: {min_length_mm} mm")
    
    # 2. Validation & Loading
    if not lengths_file.exists():
        raise FileNotFoundError(f"Lengths file not found: {lengths_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    if not pred_file.exists():
        print(f"Warning: Predictions file not found: {pred_file}")
        pred_labels = None
    else:
        print(f"Loading predictions from {pred_file}...")
        # Predictions usually don't have header, but load_labels_file handles both
        pred_labels = load_labels_file(pred_file)
        
    print("Loading lengths...")
    lengths = fast_load_txt(str(lengths_file))
    
    print("Loading labels...")
    labels = load_labels_file(labels_file)
    
    # Check dimensions
    n_streamlines = len(lengths)
    n_labels = len(labels)
    
    print(f"Loaded {n_streamlines} lengths and {n_labels} labels.")
    
    if n_streamlines != n_labels:
        # Sometimes there's a mismatch if generation failed or files are out of sync
        # But user said "Both should have the same length"
        raise ValueError(f"Mismatch in lengths ({n_streamlines}) and labels ({n_labels}). Aborting.")

    if pred_labels is not None:
        if len(pred_labels) != n_streamlines:
             raise ValueError(f"Mismatch in pred labels ({len(pred_labels)}) and lengths ({n_streamlines}). Aborting.")
        print(f"Loaded {len(pred_labels)} predicted labels.")

    # 3. Filtering
    print(f"Applying filter: length >= {min_length_mm}")
    mask = lengths >= min_length_mm
    
    filtered_labels = labels[mask]
    if pred_labels is not None:
        filtered_pred_labels = pred_labels[mask]
    else:
        filtered_pred_labels = None
        
    n_kept = np.sum(mask)
    print(f"Kept {n_kept} streamlines ({n_kept/n_streamlines*100:.2f}%)")
    
    if n_kept == 0:
        print("Warning: No streamlines kept! Connectomes will be empty.")
    
    # 4. Build Connectomes
    # Expected dimensions
    roi_count = 84 if 'aparc+aseg' in atlas_name else 164
    
    # Use ConnectomeBuilder logic manually to be explicit with (u, v) pairs
    # Standard builder often takes symmetric single labels, but here we have pairs.
    
    def build_from_pairs(pairs, weights=None, n_rois=84, agg='mean'):
        mat = np.zeros((n_rois, n_rois))
        
        # Determine if 1-based indexing (0 is background/unassigned)
        df = pd.DataFrame(pairs, columns=['u', 'v'])
        
        if weights is not None:
             df['w'] = weights
       
        # Check max index to determine shift
        max_idx = df[['u', 'v']].max().max()
        if max_idx == n_rois:
            print(f"Detected 1-based indexing (max={max_idx}, n_rois={n_rois}). Shifting indices -1 and dropping 0s.")
            df['u'] = df['u'] - 1
            df['v'] = df['v'] - 1
            
            valid_mask = (df['u'] >= 0) & (df['v'] >= 0) & (df['u'] < n_rois) & (df['v'] < n_rois)
            n_dropped = len(df) - valid_mask.sum()
            if n_dropped > 0:
                 print(f"Dropped {n_dropped} connections involving unassigned (0) or out-of-bounds nodes.")
            df = df[valid_mask]
            
        else:
             print(f"Assuming 0-based indexing (max={max_idx}). No shift applied.")
             valid_mask = (df['u'] < n_rois) & (df['v'] < n_rois)
             df = df[valid_mask]

        if weights is not None:
             # Group by edge (u, v) and take mean/sum
             uv = df[['u', 'v']].values
             uv.sort(axis=1)
             df['u'] = uv[:, 0]
             df['v'] = uv[:, 1]
             
             if agg == 'sum':
                grouped = df.groupby(['u', 'v'])['w'].sum().reset_index()
             else:
                grouped = df.groupby(['u', 'v'])['w'].mean().reset_index()
             
             # Vectorized assignment
             u_idx = grouped['u'].astype(int).values
             v_idx = grouped['v'].astype(int).values
             w_vals = grouped['w'].values
             
             mat[u_idx, v_idx] = w_vals
             # Force symmetry
             mat[v_idx, u_idx] = w_vals
                     
        else:
             # NOS - just counts
             uv = df[['u', 'v']].values
             uv.sort(axis=1)
             df_pairs = pd.DataFrame(uv, columns=['u', 'v'])
             grouped = df_pairs.groupby(['u', 'v']).size().reset_index(name='count')
             
             # Vectorized assignment
             u_idx = grouped['u'].astype(int).values
             v_idx = grouped['v'].astype(int).values
             c_vals = grouped['count'].values
             
             mat[u_idx, v_idx] += c_vals
             # Handle symmetry where u != v
             off_diag_mask = u_idx != v_idx
             mat[v_idx[off_diag_mask], u_idx[off_diag_mask]] += c_vals[off_diag_mask]
                     
        return mat

    # Define Output Directory for Thresholded Results
    # Recommendation: Store in analysis folder so it's separated from raw output
    res_dir = subject_dir / "analysis" / f"{atlas_name}_minlen{int(min_length_mm)}"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # A. NOS Connectome
    print("Building NOS connectome...")
    conn_nos = build_from_pairs(filtered_labels, weights=None, n_rois=roi_count)
    
    # Save NOS
    out_nos = res_dir / f"connectome_true_nos_{atlas_name}.csv"
    pd.DataFrame(conn_nos).to_csv(out_nos, header=False, index=False)
    print(f"Saved NOS to {out_nos}")
    
    # B. FA Connectome (if available)
    if fa_file.exists():
        print("Loading FA values...")
        fa_vals = fast_load_txt(str(fa_file))
        if len(fa_vals) != n_streamlines:
            print(f"Warning: FA values length {len(fa_vals)} != streamlines {n_streamlines}. Skipping FA.")
        else:
            print("Building FA connectome...")
            filtered_fa = fa_vals[mask]
            conn_fa = build_from_pairs(filtered_labels, weights=filtered_fa, n_rois=roi_count)
            out_fa = res_dir / f"connectome_true_fa_{atlas_name}.csv"
            pd.DataFrame(conn_fa).to_csv(out_fa, header=False, index=False)
            print(f"Saved FA to {out_fa}")
    else:
        print("FA file not found, skipping.")

    # C. SIFT2 Connectome (if available)
    # SIFT2 weights are summed, not averaged (unlike FA) - wait,
    # Standard SIFT2 connectome is sum of weights of streamlines connecting nodes.
    # Check visualization: 'sift2': {'title': 'SIFT2', 'log_scale': True}
    # Usually SIFT2 replaces "count". So it's sum.
    
    if sift_file.exists():
        print("Loading SIFT2 weights...")
        sift_vals = fast_load_txt(str(sift_file))
        if len(sift_vals) != n_streamlines:
             print(f"Warning: SIFT values length {len(sift_vals)} != streamlines {n_streamlines}. Skipping SIFT2.")
        else:
            print("Building SIFT2 connectome...")
            filtered_sift = sift_vals[mask]
            
            # Use updated build_from_pairs with sum
            mat_sift = build_from_pairs(filtered_labels, weights=filtered_sift, n_rois=roi_count, agg='sum')
            
            out_sift = res_dir / f"connectome_true_sift2_{atlas_name}.csv"
            pd.DataFrame(mat_sift).to_csv(out_sift, header=False, index=False)
            print(f"Saved SIFT2 to {out_sift}")

    # ==========================
    # 5. Build Predicted Connectomes (if labels available)
    # ==========================
    if filtered_pred_labels is not None:
        print("\n--- Building Predicted Connectomes ---")
        
        # A. Pred NOS
        print("Building Pred NOS connectome...")
        conn_pred_nos = build_from_pairs(filtered_pred_labels, weights=None, n_rois=roi_count)
        out_pred_nos = res_dir / f"connectome_pred_nos_{atlas_name}.csv"
        pd.DataFrame(conn_pred_nos).to_csv(out_pred_nos, header=False, index=False)
        print(f"Saved Pred NOS to {out_pred_nos}")
        
        # B. Pred FA
        if fa_file.exists():
            print("Building Pred FA connectome...")
            if len(fa_vals) == n_streamlines:
                filtered_fa = fa_vals[mask]
                conn_pred_fa = build_from_pairs(filtered_pred_labels, weights=filtered_fa, n_rois=roi_count)
                out_pred_fa = res_dir / f"connectome_pred_fa_{atlas_name}.csv"
                pd.DataFrame(conn_pred_fa).to_csv(out_pred_fa, header=False, index=False)
                print(f"Saved Pred FA to {out_pred_fa}")
                
        # C. Pred SIFT2
        if sift_file.exists():
             print("Building Pred SIFT2 connectome...")
             if len(sift_vals) == n_streamlines:
                filtered_sift = sift_vals[mask]
                
                # Custom SIFT Logic for Pred (same structure)
                # Need to replicate the specific SIFT-sum code for pred labels
                # Since 'build_from_pairs' uses mean for weights, we do manual here
                # Or we can refactor.
                # Actually build_from_pairs handles weights by averaging.
                # SIFT2 needs SUM.
                
                mat_pred_sift = build_from_pairs(filtered_pred_labels, weights=filtered_sift, n_rois=roi_count, agg='sum')

                out_pred_sift = res_dir / f"connectome_pred_sift2_{atlas_name}.csv"
                pd.DataFrame(mat_pred_sift).to_csv(out_pred_sift, header=False, index=False)
                print(f"Saved Pred SIFT2 to {out_pred_sift}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-id', required=True)
    parser.add_argument('--min-length', type=float, required=True, help='Minimum length in mm')
    parser.add_argument('--atlas', default='aparc+aseg', help='Atlas name (aparc+aseg or aparc.a2009s+aseg)')
    
    args = parser.parse_args()
    
    try:
        filter_and_rebuild_connectomes(args.subject_id, args.min_length, args.atlas)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
