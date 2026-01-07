import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_subject_connectomes(subject_id, base_path, output_dir):
    """
    Visualizes connectomes for one subject.
    Generates 3 types of plots:
    1. 84 ROIs (Rows: NOS, FA, SIFT2)
    2. 164 ROIs (Rows: NOS, FA, SIFT2)
    3. All combined (6 rows)
    """
    
    # Configuration
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    atlases_map = {
        '84 ROIs': 'aparc+aseg',
        '164 ROIs': 'aparc.a2009s+aseg'
    }
    
    # Map visual title to file type key (matching compute_connectome_similarities.py)
    weightings_map = {
        'nos': {'title': 'Number of streamlines-weighted', 'log_scale': True},
        'fa': {'title': 'mean FA-weighted', 'log_scale': False},
        'sift2': {'title': 'SIFT2-weighted', 'log_scale': True}
    }
    
    # Custom Colormaps
    # Jet-like (Blue->Green->Yellow->Orange->Red)
    cmap_conn = plt.get_cmap('jet')
    
    # Diverging (Blue->White->Red)
    cmap_diff = plt.get_cmap('RdBu_r')

    # FA Sensitive Colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_fa = ["#00008b", "#00ffff", "#ffff00", "#ff8c00", "#8b0000"]
    nodes_fa = [0.0, 0.35, 0.5, 0.65, 1.0] 
    cmap_fa_sensitive = LinearSegmentedColormap.from_list("FA_Sens", list(zip(nodes_fa, colors_fa)))
    
    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']

    def get_files(atlas_name, ctype):
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
        return ft, fp

    def create_grid_plot(plot_name, layout, show_atlas_label=True):
        """
        layout: list of tuples (atlas_label, atlas_name, ctype, w_info)
        """
        n_rows = len(layout)
        # Increase fig height based on rows
        fig, axes = plt.subplots(n_rows, 3, figsize=(21, 6 * n_rows), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, wspace=0.1, hspace=0.05)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        has_data = False
        
        for row_idx, (atlas_label, atlas_name, ctype, w_info) in enumerate(layout):
            ft, fp = get_files(atlas_name, ctype)
            
            if not (ft.exists() and fp.exists()):
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
            
            # Determine Scale and Colormap
            all_vals = np.concatenate([mat_t.flatten(), mat_p.flatten()])
            
            if ctype == 'fa':
                vmin, vmax, vcenter = 0.0, 1.0, 0.5
                current_cmap = cmap_fa_sensitive
                norm_conn = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            else:
                current_cmap = cmap_conn
                # NOS and SIFT2
                vmax = np.percentile(all_vals[all_vals > 0], 99) if np.any(all_vals > 0) else 1.0
                vmax = np.max(all_vals)  # Cap max
                
                if w_info['log_scale']:
                    min_pos = np.min(all_vals[all_vals > 0]) if np.any(all_vals > 0) else 0.1
                    vmin = max(min_pos, 1e-3)
                    norm_conn = mcolors.LogNorm(vmin=vmin, vmax=max(vmax, vmin+1e-3))
                    mat_t = mat_t + 1e-9
                    mat_p = mat_p + 1e-9
                else:
                    vmin = 0.0
                    norm_conn = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Difference Scaling
            max_diff = np.max(np.abs(mat_diff)) if len(mat_diff) > 0 else 1.0
            norm_diff = mcolors.Normalize(vmin=-max_diff, vmax=max_diff)
            
            # Labels
            metric_label = w_info['title']
            if show_atlas_label:
                row_title = f"{atlas_label}\n{metric_label}"
            else:
                row_title = metric_label
            
            # Plot 1: Traditional
            ax_t = axes[row_idx, 0]
            im_t = ax_t.imshow(mat_t, cmap=current_cmap, norm=norm_conn, origin='upper')
            if row_idx == 0: ax_t.set_title("Traditional connectome", fontsize=22, pad=10)
            
            # Label the row on the left
            ax_t.set_ylabel(row_title, fontsize=20, rotation=90, labelpad=20)
            # Ensure ticks are gone but label remains
            ax_t.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            
            divider_t = make_axes_locatable(ax_t)
            cax_t = divider_t.append_axes("right", size="5%", pad=0.1)
            cb_t = fig.colorbar(im_t, cax=cax_t)
            cb_t.ax.tick_params(labelsize=14)
            cb_t.set_label(metric_label, fontsize=18)

            # Plot 2: Predicted
            ax_p = axes[row_idx, 1]
            im_p = ax_p.imshow(mat_p, cmap=current_cmap, norm=norm_conn, origin='upper')
            if row_idx == 0: ax_p.set_title("Predicted connectome", fontsize=22, pad=10)
            
            ax_p.set_xticks([])
            ax_p.set_yticks([])
            
            divider_p = make_axes_locatable(ax_p)
            cax_p = divider_p.append_axes("right", size="5%", pad=0.1)
            cb_p = fig.colorbar(im_p, cax=cax_p)
            cb_p.ax.tick_params(labelsize=14)
            cb_p.set_label(metric_label, fontsize=18)

            # Plot 3: Difference
            ax_d = axes[row_idx, 2]
            im_d = ax_d.imshow(mat_diff, cmap=cmap_diff, norm=norm_diff, origin='upper')
            if row_idx == 0: ax_d.set_title("Difference map", fontsize=22, pad=10)
            
            ax_d.set_xticks([])
            ax_d.set_yticks([])
            
            divider_d = make_axes_locatable(ax_d)
            cax_d = divider_d.append_axes("right", size="5%", pad=0.1)
            cb_d = fig.colorbar(im_d, cax=cax_d)
            diff_label = f"Difference in\n{metric_label}"
            cb_d.set_label(diff_label, fontsize=18)
            cb_d.ax.tick_params(labelsize=14)

        if has_data:
            out_file = output_dir / f"{subject_id}_{plot_name}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {out_file}")
            plt.close(fig)
        else:
            plt.close(fig)

    # Define Layouts
    ordered_ctypes = ['nos', 'fa', 'sift2']
    
    # 1. 84 ROIs
    label_84 = '84 ROIs'
    layout_84 = []
    for ctype in ordered_ctypes:
        layout_84.append((label_84, atlases_map[label_84], ctype, weightings_map[ctype]))
    
    print(f"Generating 84 ROIs plot...")
    create_grid_plot("connectomes_84ROIs", layout_84, show_atlas_label=False)
    
    # 2. 164 ROIs
    label_164 = '164 ROIs'
    layout_164 = []
    for ctype in ordered_ctypes:
        layout_164.append((label_164, atlases_map[label_164], ctype, weightings_map[ctype]))
    
    print(f"Generating 164 ROIs plot...")
    create_grid_plot("connectomes_164ROIs", layout_164, show_atlas_label=False)
    
    # 3. Combined
    print(f"Generating Combined plot...")
    layout_combined = layout_84 + layout_164
    create_grid_plot("connectomes_all", layout_combined, show_atlas_label=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_id')
    parser.add_argument('--base-dir', default='/media/volume/MV_HCP')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()
    
    visualize_subject_connectomes(args.subject_id, args.base_dir, args.out_dir)
