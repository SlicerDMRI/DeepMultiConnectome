import pandas as pd
import numpy as np
import os
import pingouin as pg

# Load the dataset
cwd = os.getcwd()
csv_path = os.path.join(cwd, "data/TRT_combined_aggregated_metrics.csv")
data = pd.read_csv(csv_path)

# Map atlas names
atlas_mapping = {"aparc+aseg": "84 ROIs", "aparc.a2009s+aseg": "164 ROIs"}
data['atlas'] = data['atlas'].replace(atlas_mapping)

# Define graph measures
measures = ['Modularity', 'Clustering Coefficient', 'Path Length', 'Global Efficiency', 'Local Efficiency',
            'Assortativity', 'Global Reaching Centrality', 'Network Density']

# Initialize results
icc_cv_results = []

# Create output paths
output_dir = os.path.join(cwd, "results")
os.makedirs(output_dir, exist_ok=True)

# Process each atlas
for atlas in data['atlas'].unique():
    atlas_data = data[data['atlas'] == atlas]
    
    # Compute ICC for each measure
    for measure in measures:
        for version in ['true', 'pred']:  # Renamed 'state' to 'version'
            measure_data = atlas_data[["mode", "subject_id", f"{measure} {version}"]]
            
            # Compute ICC
            icc = pg.intraclass_corr(data=measure_data, targets='subject_id', raters='mode', ratings=f"{measure} {version}")
            icc.set_index('Type', inplace=True)
            
            # Extract ICC value, p-value, and confidence intervals
            icc_value = icc.loc['ICC3', 'ICC']  # Get ICC(3,1) from the result
            p_value = icc.loc['ICC3', 'pval']
            ci_low, ci_high = icc.loc['ICC3', 'CI95%']  # Get 95% confidence interval

            # Compute CV% for the true and predicted versions
            cv_version = measure_data[f"{measure} {version}"].std() / measure_data[f"{measure} {version}"].mean() * 100
            
            # Add to results
            icc_cv_results.append({
                'Atlas': atlas,
                'Measure': measure,
                'Version': version,  # Renamed 'State' to 'Version'
                'ICC': icc_value,
                'p-value': p_value,
                'CI95% Low': ci_low,
                'CI95% High': ci_high,
                'CV%': cv_version
            })

# Save results to CSV
icc_cv_results_df = pd.DataFrame(icc_cv_results)
icc_cv_results_df.to_csv(os.path.join(output_dir, "icc_cv_results.csv"), index=False)

# Print completion message
print(f"Analysis completed. Results saved in {output_dir}.")


import seaborn as sns
import matplotlib.pyplot as plt

# Create the output directory
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Load the results
icc_cv_results_df = pd.read_csv(os.path.join(output_dir, "icc_cv_results.csv"))

# 1. Heatmap of ICC Values
# for atlas in icc_cv_results_df['Atlas'].unique():
#     atlas_data = icc_cv_results_df[icc_cv_results_df['Atlas'] == atlas]
#     pivot = atlas_data.pivot(index="Measure", columns="Version", values="ICC")

#     plt.figure(figsize=(10, 6))
#     sns.heatmap(pivot, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'ICC'})
#     plt.title(f"ICC Heatmap - {atlas}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_dir, f"icc_heatmap_{atlas}.png"))
#     plt.close()

# # 2. Bar Plot of ICC Values
# plt.figure(figsize=(12, 8))
# sns.barplot(data=icc_cv_results_df, x="Measure", y="ICC", hue="Version")
# plt.xticks(rotation=45, ha="right")
# plt.title("ICC Values by Measure and Version")
# plt.ylabel("ICC")
# plt.legend(title="Version")
# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "icc_barplot.png"))
# plt.close()

# # 3. CV% Comparison Bar Plot
# plt.figure(figsize=(12, 8))
# sns.barplot(data=icc_cv_results_df, x="Measure", y="CV%", hue="Version")
# plt.xticks(rotation=45, ha="right")
# plt.title("CV% by Measure and Version")
# plt.ylabel("Coefficient of Variation (%)")
# plt.legend(title="Version")
# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "cv_comparison_barplot.png"))
# plt.close()

# # 4. Confidence Interval Ranges
# plt.figure(figsize=(12, 8))
# for atlas in icc_cv_results_df['Atlas'].unique():
#     atlas_data = icc_cv_results_df[icc_cv_results_df['Atlas'] == atlas]
#     plt.errorbar(atlas_data['Measure'], atlas_data['ICC'],
#                  yerr=[atlas_data['ICC'] - atlas_data['CI95% Low'], 
#                        atlas_data['CI95% High'] - atlas_data['ICC']],
#                  fmt='o', label=atlas)
# plt.xticks(rotation=45, ha="right")
# plt.title("ICC with Confidence Intervals")
# plt.ylabel("ICC")
# plt.legend(title="Atlas")
# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "icc_confidence_intervals.png"))
# plt.close()
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Times New Roman font is used
# plt.rcParams["font.family"] = "Times New Roman"

# Define the plot directory
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Define the custom measure order
measures = ['Modularity', 'Clustering Coefficient', 'Path Length', 'Global Efficiency', 'Local Efficiency',
            'Assortativity']#, 'Global Reaching Centrality', 'Network Density']

# Filter data for the two atlases
heatmap_data = icc_cv_results_df.pivot_table(
    index="Measure", columns=["Atlas", "Version"], values="ICC"
)
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1]}, dpi=200)

# Define atlases and iterate over axes
atlases = ["84 ROIs", "164 ROIs"]
for i, atlas in enumerate(atlases):
    # Filter data for the current atlas
    atlas_data = icc_cv_results_df[icc_cv_results_df['Atlas'] == atlas]
    pivot = atlas_data.pivot(index="Measure", columns="Version", values="ICC").reindex(measures)
    
    # Rename and reorder x-axis labels
    pivot = pivot.rename(columns={"true": "Ground Truth", "pred": "Predicted"})
    pivot = pivot[["Ground Truth", "Predicted"]]  # Reorder columns

    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=i == 1,  # Show color bar only on the second subplot
        ax=axes[i],
        cbar_kws={'label': 'ICC'},
        square=True  # Make blocks square
    )
    axes[i].set_title(atlas, fontsize=14)
    axes[i].set_xlabel("")  # Remove x-axis label
    axes[i].set_ylabel("")  # Remove y-axis label
    if i == 0:
        axes[i].tick_params(axis='y', labelsize=10)  # Only show y-axis ticks on the first subplot

# Adjust layout and save the figure
fig.suptitle("ICC Heatmaps for 84 ROIs and 164 ROIs", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "icc_heatmap.png"))
plt.close()

print(f"Combined heatmap saved to {os.path.join(plot_dir, 'icc_heatmap.png')}")
