import os
import pandas as pd
import numpy as np
from itertools import combinations

# Load subject IDs
subject_file = '/media/volume/MV_HCP/subjects_tractography_output_1000_test.txt'
with open(subject_file, 'r') as f:
    subject_ids = [line.strip() for line in f.readlines()]

# Atlases to process
atlases = ["aparc+aseg", "aparc.a2009s+aseg"]

# Function to extract upper triangular values
def upper_triangle(matrix):
    return matrix[np.triu_indices_from(matrix, k=0)]  # k=0 includes diagonal

# Function to compute NMAE
def compute_mae_edge(pred_upper, true_upper):
    """
    Compute the Mean Absolute Error (MAE) at the edge level between two connectomes.

    This function calculates the average absolute difference between corresponding
    elements (edges) in the upper triangular portions of the predicted and ground
    truth connectome matrices. It provides a pixel- or edge-wise error measure.

    Example:
        If pred_upper = [1, 2, 3] and true_upper = [1, 3, 5],
        MAE = mean(abs([1 - 1, 2 - 3, 3 - 5])) = (0 + 1 + 2) / 3 = 1.0
    """
    return np.mean(np.abs(pred_upper - true_upper))


def compute_nmae(pred_upper, true_upper):
    """
    Compute the Normalized Mean Absolute Error (NMAE) for connectomes.

    This function calculates the absolute error between the predicted and ground truth
    connectomes, normalized by the total sum of the ground truth values. It provides a
    global error measure that takes into account the overall magnitude of the ground
    truth connectome.

    Example:
        If pred_upper = [1, 2, 3] and true_upper = [1, 3, 5],
        Total absolute error = sum(abs([1 - 1, 2 - 3, 3 - 5])) = 0 + 1 + 2 = 3
        Sum of true_upper = 1 + 3 + 5 = 9
        NMAE = Total absolute error / Sum of true_upper = 3 / 9 = 0.333
    """
    return np.sum(np.absolute(true_upper - pred_upper)) / np.sum(true_upper)

# Function to process intrasubject metrics
def compute_intrasubject_metrics(subject_id, atlas):
    pred_path = f'/media/volume/MV_HCP/HCP_MRtrix/{subject_id}/TractCloud/connectome_{atlas}_pred.csv'
    true_path = f'/media/volume/MV_HCP/HCP_MRtrix/{subject_id}/TractCloud/connectome_{atlas}_true.csv'

    if os.path.exists(pred_path) and os.path.exists(true_path):
        pred_matrix = pd.read_csv(pred_path, header=None).values
        true_matrix = pd.read_csv(true_path, header=None).values

        pred_upper = upper_triangle(pred_matrix)
        true_upper = upper_triangle(true_matrix)

        correlation = np.corrcoef(pred_upper, true_upper)[0, 1]
        mae_edge = compute_mae_edge(pred_upper, true_upper)
        nmae = compute_nmae(pred_upper, true_upper)

        return {
            "subject_id": subject_id,
            "atlas": atlas,
            "intrasubject_correlation": correlation,
            "intrasubject_mae_edge": mae_edge,
            "intrasubject_nmae": nmae
        }
    else:
        print(f"Warning: Files for subject {subject_id} and atlas {atlas} not found.")
        return None

# Function to compute intersubject metrics
def compute_intersubject_metrics(atlas, true_matrices):
    intersubject_results = []
    for (sub1, mat1), (sub2, mat2) in combinations(true_matrices.items(), 2):
        mat1_upper = upper_triangle(mat1)
        mat2_upper = upper_triangle(mat2)

        correlation = np.corrcoef(mat1_upper, mat2_upper)[0, 1]
        mae_edge = compute_mae_edge(mat1_upper, mat2_upper)
        nmae = compute_nmae(mat1_upper, mat2_upper)

        intersubject_results.append({
            "subject_1": sub1,
            "subject_2": sub2,
            "atlas": atlas,
            "intersubject_correlation": correlation,
            "intersubject_mae_edge": mae_edge,
            "intersubject_nmae": nmae
        })
    return intersubject_results

# DataFrames to store results
intrasubject_df = pd.DataFrame()
intersubject_df = pd.DataFrame()

# Dictionary to store true connectomes for intersubject computation
true_connectomes = {atlas: {} for atlas in atlases}

# Process all subjects sequentially for intrasubject metrics
for atlas in atlases:
    for subject_id in subject_ids:
        result = compute_intrasubject_metrics(subject_id, atlas)
        if result is not None:
            intrasubject_df = pd.concat([intrasubject_df, pd.DataFrame([result])], ignore_index=True)
            # Store true connectomes for intersubject computation
            true_path = f'/media/volume/MV_HCP/HCP_MRtrix/{subject_id}/TractCloud/connectome_{atlas}_true.csv'
            true_connectomes[atlas][subject_id] = pd.read_csv(true_path, header=None).values

# Compute intersubject metrics for each atlas
for atlas in atlases:
    metrics = compute_intersubject_metrics(atlas, true_connectomes[atlas])
    intersubject_df = pd.concat([intersubject_df, pd.DataFrame(metrics)], ignore_index=True)

# Save results
intrasubject_df.to_csv('/media/volume/HCP_diffusion_MV/TractCloud/analysis/data/intrasubject_metrics.csv', index=False)
intersubject_df.to_csv('/media/volume/HCP_diffusion_MV/TractCloud/analysis/data/intersubject_metrics.csv', index=False)

print("Metrics computation completed.")
print(f"Intrasubject metrics saved to /media/volume/HCP_diffusion_MV/TractCloud/analysis/data/intrasubject_metrics.csv")
print(f"Intersubject metrics saved to /media/volume/HCP_diffusion_MV/TractCloud/analysis/data/intersubject_metrics.csv")

# Compute mean and std for intrasubject metrics
intrasubject_stats = intrasubject_df.groupby('atlas').agg(
    mean_correlation=('intrasubject_correlation', 'mean'),
    std_correlation=('intrasubject_correlation', 'std'),
    mean_mae_edge=('intrasubject_mae_edge', 'mean'),
    std_mae_edge=('intrasubject_mae_edge', 'std'),
    mean_nmae=('intrasubject_nmae', 'mean'),
    std_nmae=('intrasubject_nmae', 'std')
).reset_index()

# Compute mean and std for intersubject metrics
intersubject_stats = intersubject_df.groupby('atlas').agg(
    mean_correlation=('intersubject_correlation', 'mean'),
    std_correlation=('intersubject_correlation', 'std'),
    mean_mae_edge=('intersubject_mae_edge', 'mean'),
    std_mae_edge=('intersubject_mae_edge', 'std'),
    mean_nmae=('intersubject_nmae', 'mean'),
    std_nmae=('intersubject_nmae', 'std')
).reset_index()

# Format results for use in the paper
# for _, row in intrasubject_stats.iterrows():
#     print(f"{row['atlas']} - Intrasubject: Correlation {row['mean_correlation']:.3f}±{row['std_correlation']:.3f}; edge-wise-MAE {row['mean_mae_edge']:.3f}±{row['std_mae_edge']:.3f}; NMAE {row['mean_nmae']*100:.3f}±{row['std_nmae']*100:.3f}%")

# for _, row in intersubject_stats.iterrows():
#     print(f"{row['atlas']} - Intersubject: Correlation {row['mean_correlation']:.3f}±{row['std_correlation']:.3f}; edge-wise-MAE {row['mean_mae_edge']:.3f}±{row['std_mae_edge']:.3f}; NMAE {row['mean_nmae']*100:.3f}±{row['std_nmae']*100:.3f}%")


def print_table(stats, title):
    print(f"\n{title}")
    print(f"{'Atlas':<20} {'Metric':<15} {'Mean':>10} {'Std Dev':>10}")
    print("-" * 55)
    for _, row in stats.iterrows():
        print(f"{row['atlas']:<20} {'Correlation':<15} {row['mean_correlation']:.3f}±{row['std_correlation']:.3f}")
        print(f"{row['atlas']:<20} {'Edge-wise MAE':<15} {row['mean_mae_edge']:.3f}±{row['std_mae_edge']:.3f}")
        print(f"{row['atlas']:<20} {'NMAE (%)':<15} {row['mean_nmae'] * 100:.3f}±{row['std_nmae'] * 100:.3f}")
        print("-" * 55)

# Print intrasubject and intersubject stats in table format
print_table(intrasubject_stats, "Intrasubject Metrics")
print_table(intersubject_stats, "Intersubject Metrics")
