import sys
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('../../')
from utils.metrics_connectome import *
from tractography.label_encoder import *
# Define paths and atlases
cwd = os.getcwd()
subject_file = '/media/volume/MV_HCP/subjects_tractography_output_TRT.txt'
output_dir = os.path.join(cwd, 'data/')
atlases = ["aparc+aseg", "aparc.a2009s+aseg"]

# Function to process each subject's metrics
def process_subject_metrics(subject_id, atlas, mode):
    out_path = f'/media/volume/MV_HCP/HCP_MRtrix_{mode}/{subject_id}/TractCloud/'
    pred_path = f'/media/volume/MV_HCP/HCP_MRtrix_{mode}/{subject_id}/TractCloud/predictions_{atlas}_symmetric.txt'
    true_path = f'/media/volume/MV_HCP/HCP_MRtrix_{mode}/{subject_id}/output/labels_10M_{atlas}_symmetric.txt'
    if not(os.path.exists(true_path)):
        input_file = f'/media/volume/MV_HCP/HCP_MRtrix_{mode}/{subject_id}/output/labels_10M_{atlas}.txt'
        if atlas=="aparc+aseg":
            encode_labels_txt(input_file, true_path, 'symmetric', num_labels=85)
        else:
            encode_labels_txt(input_file, true_path, 'symmetric', num_labels=165)
        print(true_path)    
    
    csv_path = os.path.join(out_path, f'metrics_{atlas}.csv')
    # if os.path.exists(csv_path) and os.path.exists(f"/media/volume/MV_HCP/HCP_MRtrix_{mode}/{subject_id}/TractCloud/connectome_{atlas}_pred.csv"):
    #     print(f"Metrics CSV for subject {subject_id} and atlas {atlas} already exists. Reading...")
    #     df = pd.read_csv(csv_path)
    #     df['subject_id'] = subject_id
    #     df['atlas'] = atlas
    #     df['mode'] = mode
    #     return df
    if os.path.exists(true_path) and os.path.exists(pred_path):
        print(f"{subject_id} {atlas} {mode} processing...")
        with open(true_path, 'r') as file:
            true_labels = [int(line.strip()) for line in file]
        with open(pred_path, 'r') as file:
            pred_labels = [int(line.strip()) for line in file]

        # Compute connectome metrics
        CM = ConnectomeMetrics(true_labels=true_labels, pred_labels=pred_labels, atlas=atlas, out_path=out_path, graph=True, plot=False)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['subject_id'] = subject_id
            df['atlas'] = atlas
            df['mode'] = mode
        else:
            print(f"Warning: Metrics CSV for subject {subject_id} and atlas {atlas} not found.")
        return df
    else:
        print(f"Warning: Files for subject {subject_id} {mode} and atlas {atlas} not found.")
    return None

# Main processing function
def compute_metrics(mode):
    # Load subject IDs
    with open(subject_file, 'r') as f:
        subject_ids = [line.strip() for line in f.readlines()]

    all_metrics_df = pd.DataFrame()

    # Run computations in parallel
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(process_subject_metrics, subject_id, atlas, mode)
            for subject_id in subject_ids for atlas in atlases
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_metrics_df = pd.concat([all_metrics_df, result], ignore_index=True)

    # Save aggregated metrics
    output_file = os.path.join(output_dir, f'TRT_{mode}_aggregated_metrics.csv')
    all_metrics_df.to_csv(output_file, index=False)
    print(f"Aggregated metrics for mode '{mode}' saved to {output_file}")
    return all_metrics_df

if __name__ == "__main__":
    # Compute metrics for both modes
    test_metrics = compute_metrics("test")
    retest_metrics = compute_metrics("retest")

    # Combine test and retest metrics
    combined_metrics = pd.concat([test_metrics, retest_metrics], ignore_index=True)
    combined_output_file = os.path.join(output_dir, 'TRT_combined_aggregated_metrics.csv')
    combined_metrics.to_csv(combined_output_file, index=False)
    print(f"Combined metrics saved to {combined_output_file}")

