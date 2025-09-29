"""
This script computes and aggregates various connectome metrics for multiple subjects and atlases. 
It loads a list of subject IDs from a file, then for each subject, it computes connectome metrics 
by comparing true and predicted connectomes from different atlases (aparc+aseg and aparc.a2009s+aseg). 
The computations are performed in parallel using a thread pool to speed up processing. The results are 
collected into a Pandas DataFrame, and the aggregated metrics are saved as a CSV file for further analysis.
"""

import sys
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('../')
sys.path.append('../../')
from utils.metrics_connectome import *

# Load subject IDs
subject_file = '/media/volume/MV_HCP/subjects_tractography_output_1000_test.txt'
with open(subject_file, 'r') as f:
    subject_ids = [line.strip() for line in f.readlines()]

# Atlases to process
atlases = ["aparc+aseg", "aparc.a2009s+aseg"]

# Function to process each subject's metrics
def process_subject_metrics(subject_id, atlas):
    pred_path = f'/media/volume/MV_HCP/HCP_MRtrix/{subject_id}/TractCloud/predictions_{atlas}_symmetric.txt'
    true_path = f'/media/volume/MV_HCP/HCP_MRtrix/{subject_id}/output/labels_10M_{atlas}_symmetric.txt'
    out_path = f'/media/volume/MV_HCP/HCP_MRtrix/{subject_id}/TractCloud/'

    if os.path.exists(true_path) and os.path.exists(pred_path):
        print("{} {} wip".format(subject_id, atlas))
        # Load true and predicted labels
        with open(true_path, 'r') as file:
            true_labels = [int(line.strip()) for line in file]
        with open(pred_path, 'r') as file:
            pred_labels = [int(line.strip()) for line in file]

        # Compute connectome metrics
        CM = ConnectomeMetrics(true_labels=true_labels, pred_labels=pred_labels, atlas=atlas, out_path=out_path, graph=True, plot=False)
        
        # Load the resulting metrics CSV as a DataFrame
        csv_path = os.path.join(out_path, f'metrics_{atlas}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add subject ID as a new column for tracking
            df['subject_id'] = subject_id
            df['atlas'] = atlas
            return df
        else:
            print(f"Warning: Metrics CSV for subject {subject_id} and atlas {atlas} not found.")
        print("{} {} done".format(subject_id, atlas))
    else:
        print(f"Warning: Files for subject {subject_id} and atlas {atlas} not found.")
    return None

# DataFrame to store all metrics
all_metrics_df = pd.DataFrame()

# Run computations in parallel with up to 32 threads
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = []
    for atlas in atlases:
        for subject_id in subject_ids:
            futures.append(executor.submit(process_subject_metrics, subject_id, atlas))
    
    # Collect results as they complete
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            all_metrics_df = pd.concat([all_metrics_df, result], ignore_index=True)

# Optionally save the aggregated metrics
output_file = '/media/volume/HCP_diffusion_MV/TractCloud/analysis/connectomes/data/aggregated_metrics.csv'
all_metrics_df.to_csv(output_file, index=False)
print(f"Aggregated metrics saved to {output_file}")
