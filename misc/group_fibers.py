'''
This script processes tractography data and groups streamlines by their label pairs. 
It reads a VTK file containing tractography data, a text file with labels for each 
streamline, and writes out separate VTK files for each label pair with more than a 
specified number of streamlines.

Usage
python group_fibers.py <subject_id> <base_path> <min_fibers>

Example
python group_fibers.py 120010 /media/volume/HCP_diffusion_MV/data/ 500
'''

import numpy as np
import whitematteranalysis as wma
import sys
import os
import vtk
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

def group_and_write_fibers(fiber_array, labels, output_dir, min_fibers=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_dict = {}
    for idx, label_pair in enumerate(labels):
        label_pair = tuple(label_pair)
        if label_pair not in label_dict:
            label_dict[label_pair] = []
        label_dict[label_pair].append(idx)
    
    print(f"Total number of unique label pairs: {len(label_dict)}")

    for label_pair, indices in label_dict.items():
        print(f"Processing label pair: {label_pair}, Number of fibers: {len(indices)}")
        if len(indices) >= min_fibers:
            grouped_fibers = fiber_array.get_fibers(indices)
            pd = grouped_fibers.convert_to_polydata()
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(os.path.join(output_dir, f'fibers_label_{label_pair[0]}_{label_pair[1]}.vtk'))
            writer.SetInputData(pd)
            writer.Write()
    
    print(f"Number of label pairs with more than {min_fibers} streamlines: {count}")

def write_dict(dict_labels, min_fibers, output_dir):
    for label_pair, indices in dict_labels.items():
        print(f"Processing label pair: {label_pair}, Number of fibers: {len(indices)}")
        if len(indices) >= min_fibers:
            grouped_fibers = fiber_array.get_fibers(indices)
            pd = grouped_fibers.convert_to_polydata()
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(os.path.join(output_dir, f'fibers_label_{label_pair[0]}_{label_pair[1]}.vtk'))
            writer.SetInputData(pd)
            writer.Write()

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python group_fibers.py <subject_id> <base_path> <min_fibers>")
    #     sys.exit(1)
    # subject_id = sys.argv[1]
    # base_path = sys.argv[2]
    # min_fibers = int(sys.argv[3])

    subject_id = '698168'
    base_path = '/media/volume/HCP_diffusion_MV/data/'
    min_fibers = 15

    # output_dir = os.path.join(base_path, subject_id, "output", "tracts_sorted_per_label")
    tractography_path = os.path.join(base_path, subject_id, "output", "streamlines_MNI.vtk")
    true_path = os.path.join(base_path, subject_id, 'output/labels_aparc+aseg_symmetric.txt')
    pred_path = os.path.join(base_path, subject_id, 'TractCloud_MNI/connectome/predictions_aparc+aseg_symmetric.txt')

    # Check if the tractography and labels files exist
    if not os.path.exists(tractography_path):
        print(f"Error: Tractography file not found at {tractography_path}")
        sys.exit(1)

    if not os.path.exists(true_path):
        print(f"Error: Labels file not found at {true_path}")
        sys.exit(1)

    # Ensure the output directory exists
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    fiber_array, num_fibers = read_tractography(tractography_path)
    true_labels = convert_labels_list(read_labels(true_path, 'encoded'), "symmetric", mode='decode', num_labels=85)
    pred_labels = convert_labels_list(read_labels(pred_path, 'encoded'), "symmetric", mode='decode', num_labels=85)

    
    out_path='/media/volume/HCP_diffusion_MV/data/698168/TractCloud_MNI'
    os.mkdir(os.path.join(out_path, 'true'))
    os.mkdir(os.path.join(out_path, 'false'))
    
    label_dict = {}
    wrong_dict = {}
    for idx, true_pair in enumerate(true_labels):
        true_pair = tuple(true_pair)
        if true_pair not in label_dict:
            label_dict[true_pair] = []
        label_dict[true_pair].append(idx)
        
        # Add incorrect predictions to dict
        pred_pair = tuple(pred_labels[idx])
        if true_pair!=pred_pair:
            if pred_pair not in wrong_dict:
                wrong_dict[pred_pair] = []
            wrong_dict[pred_pair].append(idx)

        
    with open(os.path.join(out_path, 'wrong_predictions.txt'), 'w') as outfile:
        for key, value in dict(sorted(wrong_dict.items(), key=lambda item: item[1])).items():
            outfile.write(f"{key} {len(value)}\n")
    
    write_dict(label_dict, min_fibers=10, output_dir=os.path.join(out_path, 'true'))
    write_dict(wrong_dict, min_fibers=1, output_dir=os.path.join(out_path, 'false'))
    