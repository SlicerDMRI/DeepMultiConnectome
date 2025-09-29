import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_metadata(metadata_path):
    """Load metadata from a pickle file."""
    with open(metadata_path, 'rb') as file:
        metadata = pickle.load(file)
    return metadata

def plot_fibers_per_subject(metadata, output_dir):
    """Plot the number of fibers per subject."""
    plt.figure(figsize=(10, 6))
    # import pdb
    # pdb.set_trace()
    data = metadata['fibers_per_subject']
    splits = metadata['split_subjects']

    for split_name, subjects in splits.items():
        if subjects:
            split_fibers_per_subject = {subj: data[subj] for subj in subjects if subj in data}
            plt.bar(split_fibers_per_subject.keys(), split_fibers_per_subject.values(), alpha=0.6, label=split_name)
    
    plt.title('Number of Fibers per Subject')
    plt.xlabel('Subject ID')
    plt.ylabel('Number of Fibers')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fibers_per_subject.png'))
    plt.close()

def plot_fibers_per_label(metadata, output_dir, atlas='aparc+aseg', sort=False):
    """Plot the number of fibers per label."""
    plt.figure(figsize=(10, 6))
    data = metadata['fibers_per_label'][atlas]
    unique_labels, fibers_per_label = zip(*data.items())
    
    if sort:
        fibers_per_label = np.sort(fibers_per_label)
        plt.plot(unique_labels, fibers_per_label)
    else:
        plt.bar(unique_labels, fibers_per_label)

    plt.yscale('log')
    plt.ylabel('Number of Fibers')
    plt.xlabel('Label' if not sort else 'Label (not correct index)')
    plt.title('Number of Fibers per Label' + (' - Sorted' if sort else '') + atlas)
    plt.savefig(os.path.join(output_dir, 'fibers_per_label' + ('_sorted' if sort else '') + '_' +atlas + '.png'))
    plt.close()

def plot_labels_per_fiber_count(metadata, output_dir, encoding, atlas='aparc+aseg', xlim=None):
    """Plot the number of labels per fiber count."""
    aggregated_fibers_per_label = {}

    for label, count in metadata['fibers_per_label'][atlas].items():
        if label in aggregated_fibers_per_label:
            aggregated_fibers_per_label[label] += count
        else:
            aggregated_fibers_per_label[label] = count

    labels_per_fiber_count = {}
    for count in aggregated_fibers_per_label.values():
        if count in labels_per_fiber_count:
            labels_per_fiber_count[count] += 1
        else:
            labels_per_fiber_count[count] = 1

    data = labels_per_fiber_count

    # Extract the counts and corresponding number of labels
    unique_counts = sorted(data.keys())
    count_per_label = [data[count] for count in unique_counts]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(unique_counts, count_per_label, alpha=0.6)
    plt.title('Number of Labels per Fiber Count {}'.format(atlas))
    plt.xlabel('Number of Fibers per Label')
    plt.ylabel('Number of Labels')
    plt.grid(True, which="both", linestyle="-", linewidth=0.5)
    if xlim==None:
        plt.xscale('log')
        plt.savefig(os.path.join(output_dir, f'labels_per_fiber_count_{atlas}.png'))
    else:
        plt.xlim(-0.5,xlim)
        plt.savefig(os.path.join(output_dir, f'labels_per_fiber_count_limit{xlim}_{atlas}.png'))
    
    plt.close()

def compute_fiber_stats(metadata, output_dir, atlas): #! fix
    """Compute and print the average and std of fibers per label."""
    all_fibers_per_label = []
    for subj, fibers_per_label in metadata['fibers_per_label_per_subject'][atlas].items():
        all_fibers_per_label.extend(fibers_per_label.values())

    average_fibers_per_label = np.mean(all_fibers_per_label)
    std_fibers_per_label = np.std(all_fibers_per_label)

    print(f"Average fibers per label: {average_fibers_per_label:.2f}")
    print(f"Standard deviation of fibers per label: {std_fibers_per_label:.2f}")

    try:
        # Plotting the distribution of fibers per label
        plt.figure(figsize=(700, 60))
        plt.hist(all_fibers_per_label, bins=50, color='blue', alpha=0.7)
        plt.axvline(average_fibers_per_label, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {average_fibers_per_label:.2f}')
        plt.axvline(average_fibers_per_label + std_fibers_per_label, color='green', linestyle='dotted', linewidth=1.5, label=f'+1 Std: {average_fibers_per_label + std_fibers_per_label:.2f}')
        plt.axvline(average_fibers_per_label - std_fibers_per_label, color='green', linestyle='dotted', linewidth=1.5, label=f'-1 Std: {average_fibers_per_label - std_fibers_per_label:.2f}')
        plt.title(f'Distribution of Fibers per Label_{atlas}')
        plt.xlabel('Number of Fibers per Label')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stats{atlas}.png'))
        plt.close()
    except:
        pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    metadata_path = os.path.join(output_dir, 'metadata.pickle')

    metadata = load_metadata(metadata_path)

    plot_fibers_per_subject(metadata, output_dir)
    plot_fibers_per_label(metadata, output_dir, sort=False)
    plot_labels_per_fiber_count(metadata, output_dir, encoding)
    plot_labels_per_fiber_count(metadata, output_dir, encoding, 50)
    plot_labels_per_fiber_count(metadata, output_dir, encoding, 100)
