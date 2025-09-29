import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import os

cwd = os.getcwd()

# Load the CSV file
csv_path = os.path.join(cwd, "data/aggregated_metrics.csv")
data = pd.read_csv(csv_path)
atlas_mapping = {"aparc+aseg": "84 ROIs", "aparc.a2009s+aseg": "164 ROIs"}
data['atlas'] = data['atlas'].replace(atlas_mapping)

# Define the measures to analyze
measures = ['Modularity', 'Clustering Coefficient', 'Path Length', 'Global Efficiency', 'Local Efficiency',
            'Assortativity', 'Global Reaching Centrality', 'Network Density']
atlas_mapping = {"aparc+aseg": "84 ROIs", "aparc.a2009s+aseg": "164 ROIs"}

# Separate data by atlas
atlases = data['atlas'].unique()
results = []

# Output directories
output_csv_path = os.path.join(cwd, "data/correlation_results.csv")
output_plot_dir = os.path.join(cwd, "data/plots")
os.makedirs(output_plot_dir, exist_ok=True)

# Individual plots
scatter_paths = []
violin_paths = []

# Loop over each atlas and calculate the correlation, mean, and std for each measure
for atlas in atlases: 
    atlas_data = data[data['atlas'] == atlas]

    for measure in measures:
        true_values = atlas_data[f"{measure} true"]
        pred_values = atlas_data[f"{measure} pred"]
        
        # Compute Pearson correlation and p-value
        corr, p_value = pearsonr(true_values, pred_values)
        
        
        # Compute mean and standard deviation for true and predicted values
        true_mean = true_values.mean()
        true_std = true_values.std()
        pred_mean = pred_values.mean()
        pred_std = pred_values.std()
        
        # Compute normalized Mean Absolute Error (nMAE)
        nMAE = ((true_values - pred_values).abs().mean())/true_values.mean()
        
        # Append result as a dictionary
        results.append({
            'Atlas': atlas,
            'Measure': measure,
            'Correlation': corr,
            'P-Value': p_value,
            'True Mean': true_mean,
            'True Std': true_std,
            'Pred Mean': pred_mean,
            'Pred Std': pred_std,
            'nMAE': nMAE
        })
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=true_values, 
            y=pred_values, 
            alpha=0.6, 
            edgecolor="k",
            s=80  # Control marker size
        )

        # Annotate correlation and p-value
        plt.text(
            0.05, 0.95,  # Position: relative coordinates in the plot (x, y)
            f"Correlation = {corr:.2f}\nP-value = {p_value:.2e}",
            transform=plt.gca().transAxes,  # Use axis coordinates (0 to 1 range)
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white")
        )

        # Customize plot labels and title
        plt.xlabel("Ground truth")
        plt.ylabel("Predicted")
        plt.title(f"{measure} - {atlas}")
        scatter_path = os.path.join(output_plot_dir, f"{measure}_{atlas}_scatter.png")

        # Save and close the plot
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        scatter_paths.append(scatter_path)

# Combined scatter plots for all measures
plt.figure(figsize=(20, 8))
for i, measure in enumerate(measures):
    plt.subplot(2, 4, i + 1)  # Organize subplots in a 4x2 grid
    for atlas in atlases:
        atlas_data = data[data['atlas'] == atlas]
        true_values = atlas_data[f"{measure} true"]
        pred_values = atlas_data[f"{measure} pred"]
        sns.scatterplot(
            x=true_values, 
            y=pred_values, 
            label=f"{atlas}", 
            alpha=0.6, 
            s=50  # Marker size
        )
    plt.title(measure)
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.legend(loc='upper right', fontsize=8, title='Atlas')
plt.tight_layout()
combined_scatter_path = os.path.join(output_plot_dir, "All_scatter_plots.png")
plt.savefig(combined_scatter_path, dpi=300, bbox_inches='tight')
plt.close()


# Create violin plots for all measures
for measure in measures:
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=data.melt(
            id_vars=['atlas'], 
            value_vars=[f"{measure} true", f"{measure} pred"],
            var_name='Type', 
            value_name='Value'
        ), 
        x='atlas', y='Value', hue='Type', split=True, inner='quart'
    )
    plt.title(f"Distribution of {measure}")
    plt.xlabel("Atlas")
    plt.ylabel(measure)
    plt.xticks(rotation=45)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Ground truth', 'Predicted'], title='Connectome')
    violin_path = os.path.join(output_plot_dir, f"{measure}_violin.png")
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()

# Create combined violin plots for all measures
plt.figure(figsize=(20, 8))
for i, measure in enumerate(measures):
    plt.subplot(2, 4, i + 1)  # Organize subplots in a 4x2 grid
    sns.violinplot(
        data=data.melt(
            id_vars=['atlas'], 
            value_vars=[f"{measure} true", f"{measure} pred"],
            var_name='Type', 
            value_name='Value'
        ), 
        x='atlas', y='Value', hue='Type', split=True, inner='quart'
    )
    plt.title(measure)
    plt.xlabel("Atlas")
    plt.ylabel("Value")
    # plt.legend(title='Type', loc='upper right', fontsize=8, labels=['Ground truth', 'Predicted'])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Ground truth', 'Predicted'], title='Connectome')
plt.tight_layout()
combined_violin_path = os.path.join(output_plot_dir, "All_violin_plots.png")
plt.savefig(combined_violin_path, dpi=300, bbox_inches='tight')
plt.close()


# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
