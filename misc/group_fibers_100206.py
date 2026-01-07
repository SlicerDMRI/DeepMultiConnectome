'''
This script processes tractography data and groups streamlines by their label pairs for subject 100206.
It reads a VTK file containing tractography data, a text file with labels for each 
streamline, and writes out separate VTK files for each label pair with more than a 
specified number of streamlines.
'''

import numpy as np
import whitematteranalysis as wma
import sys
import os
import vtk
sys.path.append('/media/volume/HCP_diffusion_MV/DeepMultiConnectome/tractography')
from label_encoder import convert_labels_list

def read_tractography(tractography_path):
    print("Reading tractography data")
    pd_tractography = wma.io.read_polydata(tractography_path)
    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_tractography, points_per_fiber=15)
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    print(f'The number of fibers in tractography data is {feat.shape[0]}')
    return fiber_array, feat.shape[0]

def read_labels(labels_path, encoding='decoded'):
    print("Reading labels data")
    labels = []
    if encoding=='decoded':
        with open(labels_path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:  # Skip the first line
                label_pair = line.strip().split()
                labels.append((int(label_pair[0]), int(label_pair[1])))
        print(f'First few labels: {labels[:5]}')
    elif encoding=='encoded':
        with open(labels_path, 'r') as file:
            labels = [line.strip() for line in file]
    return labels

def write_dict(fiber_array, dict_labels, min_fibers, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    for label_pair, indices in dict_labels.items():
        if len(indices) >= min_fibers:
            print(f"Processing label pair: {label_pair}, Number of fibers: {len(indices)}")
            grouped_fibers = fiber_array.get_fibers(indices)
            pd = grouped_fibers.convert_to_polydata()
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(os.path.join(output_dir, f'fibers_label_{label_pair[0]}_{label_pair[1]}.vtk'))
            writer.SetInputData(pd)
            writer.Write()
            count += 1
    
    print(f"Number of label pairs with >= {min_fibers} streamlines: {count}")

if __name__ == "__main__":
    subject_id = '100206'
    base_path = '/media/volume/MV_HCP/HCP_MRtrix/'
    min_fibers = 15

    tractography_path = os.path.join(base_path, subject_id, "output", "streamlines_10M.vtk")
    true_path = os.path.join(base_path, subject_id, 'output/labels_10M_aparc+aseg_symmetric.txt')
    pred_path = os.path.join(base_path, subject_id, 'TractCloud/predictions_aparc+aseg_symmetric.txt')

    # Check if the tractography and labels files exist
    if not os.path.exists(tractography_path):
        print(f"Error: Tractography file not found at {tractography_path}")
        sys.exit(1)

    if not os.path.exists(true_path):
        print(f"Error: True labels file not found at {true_path}")
        sys.exit(1)
    
    if not os.path.exists(pred_path):
        print(f"Error: Prediction labels file not found at {pred_path}")
        sys.exit(1)

    # Output directory
    out_path = '/media/volume/HCP_diffusion_MV/fiber_predictions/100206'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    print(f"\nProcessing subject {subject_id}")
    print(f"Tractography: {tractography_path}")
    print(f"True labels: {true_path}")
    print(f"Pred labels: {pred_path}")
    print(f"Output: {out_path}\n")

    # Read data
    fiber_array, num_fibers = read_tractography(tractography_path)
    print(f"\nReading true labels...")
    true_labels = convert_labels_list(read_labels(true_path, 'encoded'), "symmetric", mode='decode', num_labels=85)
    print(f"\nReading predicted labels...")
    pred_labels = convert_labels_list(read_labels(pred_path, 'encoded'), "symmetric", mode='decode', num_labels=85)

    # Create output subdirectories
    true_dir = os.path.join(out_path, 'true')
    false_dir = os.path.join(out_path, 'false')
    
    if not os.path.exists(true_dir):
        os.makedirs(true_dir)
    if not os.path.exists(false_dir):
        os.makedirs(false_dir)
    
    # Group fibers by label pairs
    print("\nGrouping fibers by label pairs...")
    label_dict = {}
    wrong_dict = {}
    
    for idx, true_pair in enumerate(true_labels):
        true_pair = tuple(true_pair)
        if true_pair not in label_dict:
            label_dict[true_pair] = []
        label_dict[true_pair].append(idx)
        
        # Add incorrect predictions to dict
        pred_pair = tuple(pred_labels[idx])
        if true_pair != pred_pair:
            if pred_pair not in wrong_dict:
                wrong_dict[pred_pair] = []
            wrong_dict[pred_pair].append(idx)

    print(f"Total unique true label pairs: {len(label_dict)}")
    print(f"Total unique incorrect predictions: {len(wrong_dict)}")
    
    # Write wrong predictions summary
    wrong_predictions_file = os.path.join(out_path, 'wrong_predictions.txt')
    with open(wrong_predictions_file, 'w') as outfile:
        outfile.write(f"# Wrong predictions for subject {subject_id}\n")
        outfile.write(f"# Format: (label1, label2) count\n")
        for key, value in sorted(wrong_dict.items(), key=lambda item: len(item[1]), reverse=True):
            outfile.write(f"{key} {len(value)}\n")
    
    print(f"\nWrote wrong predictions summary to: {wrong_predictions_file}")
    
    # Write VTK files for correct predictions
    print(f"\nWriting VTK files for correct predictions (min {min_fibers} fibers)...")
    write_dict(fiber_array, label_dict, min_fibers=min_fibers, output_dir=true_dir)
    
    # Write VTK files for incorrect predictions
    print(f"\nWriting VTK files for incorrect predictions (min 1 fiber)...")
    write_dict(fiber_array, wrong_dict, min_fibers=1, output_dir=false_dir)
    
    print(f"\n✓ Processing complete!")
    print(f"Results saved to: {out_path}")
    print(f"  - Correct predictions: {true_dir}")
    print(f"  - Incorrect predictions: {false_dir}")
    print(f"  - Summary: {wrong_predictions_file}")
