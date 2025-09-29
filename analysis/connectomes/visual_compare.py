import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# Set font style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20  # Larger font size

subject_id = "965367"
path_csvs = "D:/code/ExternshipLocal/results/connectomes/"

# File paths
files = {
    "aparc+aseg": {
        "true": f"{path_csvs}/{subject_id}_connectome_aparc+aseg_true.csv",
        "pred": f"{path_csvs}/{subject_id}_connectome_aparc+aseg_pred.csv"
    },
    "aparc.a2009s+aseg": {
        "true": f"{path_csvs}/{subject_id}_connectome_aparc.a2009s+aseg_true.csv",
        "pred": f"{path_csvs}/{subject_id}_connectome_aparc.a2009s+aseg_pred.csv"
    }
}

# Load connectomes
connectomes = {}
for atlas, paths in files.items():
    connectomes[atlas] = {
        "true": pd.read_csv(paths["true"], header=None).values,
        "pred": pd.read_csv(paths["pred"], header=None).values
    }
    connectomes[atlas]["diff"] = connectomes[atlas]["true"] - connectomes[atlas]["pred"]

# Plot function with optional ticks
def plot_matrix(ax, matrix, title, difference=False, log_scale=False, show_ticks=False):
    if difference == True:
        if not log_scale:
            cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
            norm = mcolors.TwoSlopeNorm(
                vmin=matrix.min(), vcenter=0, vmax=matrix.max())
        if log_scale:
            cmap = plt.get_cmap('Reds')  # Red-white-blue colormap
            norm = mcolors.LogNorm(
                vmin=max(matrix.min(), 1), vmax=matrix.max())
            matrix = np.where(
                matrix == 0, 1e-6, matrix)
    elif difference == 'percent':
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    elif difference == 'accuracy':
        cmap = plt.get_cmap('BuGn')
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    else:
        cmap = plt.get_cmap('jet')

        if log_scale:
            norm = mcolors.LogNorm(
                vmin=max(matrix.min(), 1), vmax=matrix.max())
            # Handle zero values by replacing them with a small positive value for log scale
            matrix = np.where(
                matrix == 0, 1e-6, matrix)
        else:
            norm = None

    im = ax.imshow(matrix, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if difference:
        cbar.set_label('Difference in streamlines', fontsize=20)
    else:
        cbar.set_label('Number of streamlines', fontsize=20)

    ax.set_title(title, fontsize=33)

    # Show/hide ticks
    if show_ticks:
        num_nodes = matrix.shape[0]
        ax.set_xticks(np.arange(9, num_nodes, 10))
        ax.set_xticklabels(np.arange(10, num_nodes, 10), fontsize=14)
        ax.set_yticks(np.arange(9, num_nodes, 10))
        ax.set_yticklabels(np.arange(10, num_nodes, 10), fontsize=14)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

# Set tick visibility
show_ticks = True  # Change to True if you want ticks

# Plot all subplots
for i, (atlas, data) in enumerate(connectomes.items()):
    plot_matrix(axes[i, 0], data["true"], f"Traditional connectome", log_scale=True, show_ticks=show_ticks)
    plot_matrix(axes[i, 1], data["pred"], f"Predicted connectome", log_scale=True, show_ticks=show_ticks)
    plot_matrix(axes[i, 2], data["diff"], f"Difference map", difference=True, show_ticks=show_ticks)

plt.tight_layout()
plt.savefig(f"connectomes_subject{subject_id}.png", dpi=1000, bbox_inches="tight")
plt.show()
