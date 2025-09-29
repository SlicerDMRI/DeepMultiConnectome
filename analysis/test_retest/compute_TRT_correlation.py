import os
import pandas as pd
import numpy as np
from itertools import combinations
cwd = os.getcwd()

# Load subject IDs
subject_file = '/media/volume/MV_HCP/subjects_tractography_output_TRT.txt'
with open(subject_file, 'r') as f:
    subject_ids = [line.strip() for line in f.readlines()]

# Atlases to process
atlases = ["aparc+aseg", "aparc.a2009s+aseg"]

# Data Paths
def get_path(mode, state, subject_id, atlas):
    return f"/media/volume/MV_HCP/HCP_MRtrix_{mode}/{subject_id}/TractCloud/connectome_{atlas}_{state}.csv"

# Function to extract upper triangular values (exclude diagonal)
def upper_triangle(matrix):
    return matrix[np.triu_indices_from(matrix, k=0)]  # Excludes diagonal with k=1

# Function to compute intrasubject correlation
def compute_intrasubject_correlation(subject_id, atlas, state):
    test_path = get_path("test", state, subject_id, atlas)
    retest_path = get_path("retest", state, subject_id, atlas)

    if os.path.exists(test_path) and os.path.exists(retest_path):
        test_matrix = pd.read_csv(test_path, header=None).values
        retest_matrix = pd.read_csv(retest_path, header=None).values

        test_upper = upper_triangle(test_matrix)
        retest_upper = upper_triangle(retest_matrix)

        correlation = np.corrcoef(test_upper, retest_upper)[0, 1]
        return {"subject_id": subject_id, "atlas": atlas, "state": state, "intrasubject_correlation": correlation}
    else:
        print(f"Warning: Files for subject {subject_id}, atlas {atlas}, state {state} not found.")
        return None

# Function to compute intersubject correlations
def compute_intersubject_correlations(atlas, mode, state, matrices):
    intersubject_correlations = []
    for (sub1, mat1), (sub2, mat2) in combinations(matrices.items(), 2):
        mat1_upper = upper_triangle(mat1)
        mat2_upper = upper_triangle(mat2)
        correlation = np.corrcoef(mat1_upper, mat2_upper)[0, 1]
        intersubject_correlations.append({
            "subject_1": sub1, "subject_2": sub2, "atlas": atlas, "mode": mode, "state": state, "correlation": correlation
        })
    return intersubject_correlations

# DataFrames to store results
intrasubject_df = pd.DataFrame()
intersubject_df = pd.DataFrame()

# Process all subjects and states
for atlas in atlases:
    for state in ["pred", "true"]:
        # Dictionary to store connectomes for intersubject correlation
        connectomes = {mode: {} for mode in ["test", "retest"]}

        for mode in ["test", "retest"]:
            for subject_id in subject_ids:
                path = get_path(mode, state, subject_id, atlas)
                if os.path.exists(path):
                    connectomes[mode][subject_id] = pd.read_csv(path, header=None).values

        # Compute intrasubject correlations (test vs. retest)
        for subject_id in subject_ids:
            result = compute_intrasubject_correlation(subject_id, atlas, state)
            if result:
                intrasubject_df = pd.concat([intrasubject_df, pd.DataFrame([result])], ignore_index=True)

        # Compute intersubject correlations within each mode
        for mode in ["test", "retest"]:
            correlations = compute_intersubject_correlations(atlas, mode, state, connectomes[mode])
            intersubject_df = pd.concat([intersubject_df, pd.DataFrame(correlations)], ignore_index=True)

# Save results
output_dir = '/media/volume/HCP_diffusion_MV/TractCloud/analysis/data'
os.makedirs(output_dir, exist_ok=True)

intrasubject_df.to_csv(os.path.join(output_dir, 'intrasubject_correlation.csv'), index=False)
intersubject_df.to_csv(os.path.join(output_dir, 'intersubject_correlation.csv'), index=False)

print("Correlation computation completed.")
print(f"Intrasubject correlations saved to {os.path.join(output_dir, 'intrasubject_correlation.csv')}")
print(f"Intersubject correlations saved to {os.path.join(output_dir, 'intersubject_correlation.csv')}")

# Compute mean and std for intrasubject correlations
intrasubject_stats = intrasubject_df.groupby(['atlas', 'state'])['intrasubject_correlation'].agg(['mean', 'std']).reset_index()

# Compute mean and std for intersubject correlations
intersubject_stats = intersubject_df.groupby(['atlas', 'mode', 'state'])['correlation'].agg(['mean', 'std']).reset_index()

# Print formatted results
print("\n--- Results ---")
for _, row in intrasubject_stats.iterrows():
    print(f"{row['atlas']} ({row['state']}) - Intrasubject (mean±std): {row['mean']:.3f}±{row['std']:.3f}")

for _, row in intersubject_stats.iterrows():
    print(f"{row['atlas']} ({row['mode']}, {row['state']}) - Intersubject (mean±std): {row['mean']:.3f}±{row['std']:.3f}")

#################################################
##################### plots #####################
#################################################
import seaborn as sns
import matplotlib.pyplot as plt

# Plot directory
plot_dir = '/media/volume/HCP_diffusion_MV/TractCloud/analysis/test_retest/plots'
os.makedirs(plot_dir, exist_ok=True)

# --- Intrasubject Correlations ---
plt.figure(figsize=(10, 6))
sns.barplot(
    data=intrasubject_stats,
    x="atlas",
    y="mean",
    hue="state",
    ci=None,
    palette="viridis"
)
for i, row in intrasubject_stats.iterrows():
    plt.errorbar(
        x=i, y=row["mean"], yerr=row["std"], fmt="none", color="black", capsize=5
    )

plt.title("Intrasubject Correlations", fontsize=16)
plt.ylabel("Mean Correlation ± STD")
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="State", fontsize=12, title_fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "intrasubject_correlation.png"))
plt.close()

# --- Intersubject Correlations ---
plt.figure(figsize=(12, 6))
sns.barplot(
    data=intersubject_stats,
    x="atlas",
    y="mean",
    hue="state",
    ci=None,
    palette="magma"
)
for i, row in intersubject_stats.iterrows():
    plt.errorbar(
        x=i, y=row["mean"], yerr=row["std"], fmt="none", color="black", capsize=5
    )

plt.title("Intersubject Correlations", fontsize=16)
plt.ylabel("Mean Correlation ± STD")
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="State", fontsize=12, title_fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "intersubject_correlation.png"))
plt.close()

print(f"Plots saved in {plot_dir}")

#################################################
############### statistical tests ###############
#################################################
if False:
    from scipy.stats import ttest_rel, wilcoxon

    # Compute mean and std for intrasubject correlations
    intrasubject_stats = intrasubject_df.groupby(['atlas', 'state'])['intrasubject_correlation'].agg(['mean', 'std']).reset_index()

    # Compute mean and std for intersubject correlations
    intersubject_stats = intersubject_df.groupby(['atlas', 'mode', 'state'])['correlation'].agg(['mean', 'std']).reset_index()

    # Perform paired statistical test for intrasubject correlations (pred vs. true)
    stat_test_results = []
    for atlas in atlases:
        pred_values = intrasubject_df[
            (intrasubject_df['atlas'] == atlas) & (intrasubject_df['state'] == 'pred')
        ]['intrasubject_correlation']
        
        true_values = intrasubject_df[
            (intrasubject_df['atlas'] == atlas) & (intrasubject_df['state'] == 'true')
        ]['intrasubject_correlation']
        
        # Ensure both have the same subjects for pairing
        paired_pred = pred_values.reset_index(drop=True)
        paired_true = true_values.reset_index(drop=True)

        if len(paired_pred) == len(paired_true):
            # Paired t-test
            t_stat, t_pval = ttest_rel(paired_pred, paired_true)
            # Wilcoxon signed-rank test
            wilcoxon_stat, wilcoxon_pval = wilcoxon(paired_pred, paired_true)
            
            stat_test_results.append({
                "atlas": atlas,
                "t_stat": t_stat,
                "t_pval": t_pval,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_pval": wilcoxon_pval
            })
        else:
            print(f"Warning: Mismatch in paired data length for atlas {atlas}.")

    # Convert test results to DataFrame and print
    stat_test_df = pd.DataFrame(stat_test_results)
    print("\n--- Paired Statistical Test Results ---")
    print(stat_test_df)

    # Save the statistical test results
    stat_test_output_path = os.path.join(output_dir, 'intrasubject_statistical_tests.csv')
    stat_test_df.to_csv(stat_test_output_path, index=False)
    print(f"Statistical test results saved to {stat_test_output_path}")

    # Compute effect sizes (Cohen's d) for each atlas
    effect_sizes = []
    for atlas in atlases:
        pred_values = intrasubject_df[
            (intrasubject_df['atlas'] == atlas) & (intrasubject_df['state'] == 'pred')
        ]['intrasubject_correlation']
        
        true_values = intrasubject_df[
            (intrasubject_df['atlas'] == atlas) & (intrasubject_df['state'] == 'true')
        ]['intrasubject_correlation']
        
        paired_pred = pred_values.reset_index(drop=True)
        paired_true = true_values.reset_index(drop=True)
        
        if len(paired_pred) == len(paired_true):
            # Compute Cohen's d
            mean_diff = np.mean(paired_pred - paired_true)
            pooled_std = np.sqrt((np.std(paired_pred, ddof=1) ** 2 + np.std(paired_true, ddof=1) ** 2) / 2)
            effect_size = mean_diff / pooled_std
            effect_sizes.append({"atlas": atlas, "effect_size": effect_size})
        else:
            print(f"Warning: Mismatch in paired data length for atlas {atlas}.")

    effect_size_df = pd.DataFrame(effect_sizes)
    print("\n--- Effect Sizes (Cohen's d) ---")
    print(effect_size_df)

    # Visualization: Paired differences and effect sizes
    fig, axes = plt.subplots(1, len(atlases), figsize=(12, 6), sharey=True)
    for i, atlas in enumerate(atlases):
        pred_values = intrasubject_df[
            (intrasubject_df['atlas'] == atlas) & (intrasubject_df['state'] == 'pred')
        ]['intrasubject_correlation']
        
        true_values = intrasubject_df[
            (intrasubject_df['atlas'] == atlas) & (intrasubject_df['state'] == 'true')
        ]['intrasubject_correlation']
        
        paired_pred = pred_values.reset_index(drop=True)
        paired_true = true_values.reset_index(drop=True)
        
        if len(paired_pred) == len(paired_true):
            differences = paired_pred - paired_true
            axes[i].scatter(range(len(differences)), differences, alpha=0.7, label='Differences')
            axes[i].hlines(0, 0, len(differences), colors='red', linestyles='dashed', label='Zero Line')
            axes[i].set_title(f"{atlas} Differences\nCohen's d: {effect_sizes[i]['effect_size']:.3f}")
            axes[i].set_xlabel("Subject")
            axes[i].set_ylabel("Correlation Difference")
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cwd, "data/blant_plot.png"))
