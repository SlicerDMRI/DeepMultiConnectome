import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_subject_connectomes(subject_id, base_path, output_dir):
    """
    Visualizes traditional, predicted, and difference connectomes for one subject.
    Generates 3 plots (one for each weighting: NOS, FA, SIFT2).
    Each plot has 6 subplots (2 rows x 3 columns) for 84 vs 164 ROIs.
    """
    
    # Configuration
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    atlases = {
        '84 ROIs': 'aparc+aseg',
        '164 ROIs': 'aparc.a2009s+aseg'
    }
    
    # Map visual title to file type key (matching compute_connectome_similarities.py)
    weightings = {
        'nos': {'title': 'Number of Streamlines', 'log_scale': True},
        'fa': {'title': 'FA', 'log_scale': False},
        'sift2': {'title': 'SIFT2 weight', 'log_scale': True}
    }
    
    # Custom Colormaps
    # Jet-like (Blue->Green->Yellow->Orange->Red)
    cmap_conn = plt.get_cmap('jet')
    #cmap_conn.set_bad(color='Blue')
    
    # Diverging (Blue->White->Red)
    cmap_diff = plt.get_cmap('RdBu_r')
    
    # Font settings
    #plt.rcParams['font.family'] = 'Liberation Sans'
    #plt.rcParams.update({
    #    'font.family': 'Liberation Sans',
    #    'font.sans-serif': ['Liberation Sans'],
    #    'axes.unicode_minus': False  # Recommended for DejaVu to handle math symbols correctly
    #})
    # Instead of a single string, provide a list of preferences
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
    
    for ctype, w_info in weightings.items():
        print(f"Generating plot for {ctype}...")
        
        # Increase fig width to accommodate colorbars on the right of each subplot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
        # Increase spacing between subplots (wspace) for the labels
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, wspace=0.1, hspace=0.05)
        
        has_data = False

        # Rows: 84 ROIs, 164 ROIs
        for row_idx, (roi_label, atlas_name) in enumerate(atlases.items()):
            
            # Filename Logic: connectome_{mode}_{ctype}_{atlas}.csv
            fname_true = f"connectome_true_{ctype}_{atlas_name}.csv"
            fname_pred = f"connectome_pred_{ctype}_{atlas_name}.csv"
            
            # Look in analysis dir: HCP_MRtrix/{subject_id}/analysis/{atlas_name}
            pdir = base_path / "HCP_MRtrix" / subject_id / "analysis" / atlas_name
            
            ft = pdir / fname_true
            fp = pdir / fname_pred
            
            # Fallback checks
            if not (ft.exists() and fp.exists()):
                 ft_alt = pdir / f"{subject_id}_{fname_true}"
                 fp_alt = pdir / f"{subject_id}_{fname_pred}"
                 if ft_alt.exists() and fp_alt.exists():
                     ft, fp = ft_alt, fp_alt
                 else:
                     pdir2 = base_path / "HCP_MRtrix" / subject_id / "output"
                     ft2 = pdir2 / fname_true
                     fp2 = pdir2 / fname_pred
                     if ft2.exists() and fp2.exists():
                         ft, fp = ft2, fp2
            
            if not (ft.exists() and fp.exists()):
                # Try simple naming? e.g. connectome_{atlas}_{ctype}_true.csv
                # Just skip if not found
                # print(f"  Files not found for {atlas_name} {ctype} - Skipping row.")
                for ax in axes[row_idx]: ax.axis('off')
                continue
               
            has_data = True
            
            # Load Data
            try:
                mat_t = pd.read_csv(ft, header=None).values
                mat_p = pd.read_csv(fp, header=None).values
            except Exception as e:
                print(f"  Error loading files: {e}")
                continue
                
            # Compute Difference (True - Pred)
            mat_diff = mat_t - mat_p
            
            # Determine Scale
            all_vals = np.concatenate([mat_t.flatten(), mat_p.flatten()])
            
            if ctype == 'fa':
                vmin, vmax, vcenter = 0.0, 1.0, 0.5
                from matplotlib.colors import LinearSegmentedColormap
                colors = ["#00008b", "#00ffff", "#ffff00", "#ff8c00", "#8b0000"]
                nodes = [0.0, 0.35, 0.5, 0.65, 1.0] # This "squeezes" the yellow transition to be very tight
                cmap_fa_sensitive = LinearSegmentedColormap.from_list("FA_Sens", list(zip(nodes, colors)))
                current_cmap = cmap_fa_sensitive
                norm_conn = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            else:
                # NOS and SIFT2
                vmax = np.percentile(all_vals[all_vals > 0], 99) if np.any(all_vals > 0) else 1.0
                if w_info['log_scale']:
                    min_pos = np.min(all_vals[all_vals > 0]) if np.any(all_vals > 0) else 0.1
                    vmin = max(min_pos, 1e-3)
                    norm_conn = mcolors.LogNorm(vmin=vmin, vmax=max(vmax, vmin+1e-3))
                else:
                    vmin = 0.0
                    norm_conn = mcolors.Normalize(vmin=vmin, vmax=vmax)
                mat_t+=1e-9
                mat_p+=1e-9
            
            # Difference Scaling
            max_diff = np.max(np.abs(mat_diff)) if len(mat_diff) > 0 else 1.0
            norm_diff = mcolors.Normalize(vmin=-max_diff, vmax=max_diff)
            
            # Plot 1: Traditional
            ax_t = axes[row_idx, 0]
            im_t = ax_t.imshow(mat_t, cmap=cmap_conn, norm=norm_conn, origin='upper')
            if row_idx == 0: ax_t.set_title("Traditional Connectome", fontsize=22, pad=10)
            divider_t = make_axes_locatable(ax_t)
            cax_t = divider_t.append_axes("right", size="5%", pad=0.1)
            cb_t = fig.colorbar(im_t, cax=cax_t)
            cb_t.set_label(w_info['title'], fontsize=16)
            cb_t.ax.tick_params(labelsize=14)

            # Plot 2: Predicted
            ax_p = axes[row_idx, 1]
            im_p = ax_p.imshow(mat_p, cmap=cmap_conn, norm=norm_conn, origin='upper')
            if row_idx == 0: ax_p.set_title("Predicted Connectome", fontsize=22, pad=10)
            divider_p = make_axes_locatable(ax_p)
            cax_p = divider_p.append_axes("right", size="5%", pad=0.1)
            cb_p = fig.colorbar(im_p, cax=cax_p)
            cb_p.set_label(w_info['title'], fontsize=16)
            cb_p.ax.tick_params(labelsize=14)

            # Plot 3: Difference
            ax_d = axes[row_idx, 2]
            im_d = ax_d.imshow(mat_diff, cmap=cmap_diff, norm=norm_diff, origin='upper')
            if row_idx == 0: ax_d.set_title("Difference Map", fontsize=22, pad=10)
            divider_d = make_axes_locatable(ax_d)
            cax_d = divider_d.append_axes("right", size="5%", pad=0.1)
            cb_d = fig.colorbar(im_d, cax=cax_d)
            cb_d.set_label(f"Difference ({w_info['title']})", fontsize=16)
            cb_d.ax.tick_params(labelsize=14)

            # Remove ticks
            for ax in [ax_t, ax_p, ax_d]:
                ax.set_xticks([])
                ax.set_yticks([])

        if has_data:
            out_file = output_dir / f"{subject_id}_example_connectome_{ctype}.png"
            #fig.suptitle(f"Connectome Comparison: {subject_id} ({w_info['title']})", fontsize=20, y=1.02)
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            # print(f"Saved plot to {out_file}")
            plt.close(fig)
        else:
            plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_id')
    parser.add_argument('--base-dir', default='/media/volume/MV_HCP')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()
    
    visualize_subject_connectomes(args.subject_id, args.base_dir, args.out_dir)
