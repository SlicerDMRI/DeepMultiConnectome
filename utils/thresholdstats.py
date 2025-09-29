import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Define the number of labels for each atlas
NUM_LABELS = {"aparc+aseg": 85, "aparc.a2009s+aseg": 165}

# Function to load data from a pickle file
def load_data(root_path, split):
    file_path = os.path.join(root_path, f"{split}.pickle")
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to plot histogram of fibers per class
def plot_fibers_per_class(counts, atlas_name, output_dir):
    plt.figure(figsize=(10, 6))
    bins = np.logspace(np.log10(counts.min()), np.log10(counts.max()), 100)
    plt.hist(counts, bins=bins, log=True, color='skyblue', edgecolor='black')
    plt.xscale('log')
    plt.xlabel("Number of Fibers (log scale)")
    plt.ylabel("Classes (log scale)")
    plt.title("Distribution of Number of Fibers per Class")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    output_path = os.path.join(output_dir, f"FibersPerClass_{atlas_name.replace('.', '_')}.png")
    plt.savefig(output_path)
    plt.close()

# Function to analyze thresholds and plot results
def analyze_thresholds(atlas_labels, counts, subject_ids, thresholds, atlas_name, output_dir):
    unique_labels = np.unique(atlas_labels)
    excluded_classes = []
    excluded_fibers = []
    subject_count = len(np.unique(subject_ids))

    for threshold in thresholds:
        min_subjects_required = (threshold / 100) * subject_count
        label_subject_counts = {label: 0 for label in unique_labels}

        for subject_id in np.unique(subject_ids):
            subject_indices = np.where(subject_ids == subject_id)[0]
            subject_labels = np.unique(atlas_labels[subject_indices])
            for label in subject_labels:
                label_subject_counts[label] += 1

        rare_labels = [label for label, count in label_subject_counts.items() if count < min_subjects_required]
        num_excluded_classes = len(rare_labels)
        num_excluded_fibers = np.sum([counts[i] for i, label in enumerate(unique_labels) if label in rare_labels])

        excluded_classes.append(num_excluded_classes)
        excluded_fibers.append(num_excluded_fibers)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Threshold Level (%)")
    ax1.set_ylabel("Excluded Classes", color="tab:blue")
    ax1.plot(thresholds, excluded_classes, color="tab:blue", label="Excluded Classes")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Excluded Fibers", color="tab:red")
    ax2.plot(thresholds, excluded_fibers, color="tab:red", label="Excluded Fibers")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Impact of Thresholding on Class and Fiber Exclusion")
    fig.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    output_path = os.path.join(output_dir, f"ClassesPerThreshold_{atlas_name.replace('.', '_')}.png")
    plt.savefig(output_path)
    plt.close()

# Main execution
if __name__ == "__main__":
    SPLIT = 'val'
    ROOT_PATH = '/media/volume/MV_HCP/TrainData_MRtrix_1000_MNI_100K'
    OUTPUT_DIR = '/media/volume/HCP_diffusion_MV/TractCloud/plots'
    
    data_dict = load_data(ROOT_PATH, SPLIT)
    data_dict['label_name_aparc+aseg'] = [f"{i}_{j}" for i in range(85) for j in range(i, 85)]

    features = data_dict['feat']
    subject_ids = data_dict['subject_id']

    for atlas_name in ['aparc.a2009s+aseg']:
        atlas_labels = data_dict[f'label_{atlas_name}']
        atlas_labels = np.where(np.isin(atlas_labels, list(range(NUM_LABELS[atlas_name]))), 0, atlas_labels)
        
        unique_labels, counts = np.unique(atlas_labels, return_counts=True)

        # Plot fibers per class
        plot_fibers_per_class(counts, atlas_name, OUTPUT_DIR)

        # Analyze and plot thresholding results
        thresholds = list(range(0, 101, 1))
        analyze_thresholds(atlas_labels, counts, subject_ids, thresholds, atlas_name, OUTPUT_DIR)
