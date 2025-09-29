import os
import numpy as np
import whitematteranalysis as wma
import pickle
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from label_encoder import *
from DeepMultiConnectome.misc.format_plot import *
import sys
sys.path.append('..')
from utils.tract_feat import *

def read_tractography(tractography_dir, decay_factor):
    """Read tractography data and convert it to a feature array."""
    pd_tractography = wma.io.read_polydata(tractography_dir)
    fiber_array = CustomFiberArray()
    fiber_array.convert_from_polydata(pd_tractography, points_per_fiber=15, distribution='exponential', decay_factor=decay_factor)
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    return feat, fiber_array

class TractCloudPreprocessor:
    def __init__(self, input_dir, output_dir, subjects_file, atlases, encoding, threshold, decay_factor, split_ratios=(0.7, 0.1, 0.2), chunk_size=1):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects_file = subjects_file
        self.atlases = atlases
        self.encoding = encoding
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.split_ratios = split_ratios
        self.chunk_size = chunk_size

        os.makedirs(self.output_dir, exist_ok=True)
        
        self.num_labels={"aparc+aseg":85,
                    "aparc.a2009s+aseg":165}
        
        self.process_data()

    def read_subjects(self):
        """Read subject IDs from file to determine which subjects to process"""
        with open(self.subjects_file, 'r') as f:
            self.subjects = f.read().splitlines()
        
        self.total_subjects = len(self.subjects)
        print(f"Total number of subjects: {self.total_subjects}")

    def process_subject(self, subject_index, subject_id):
        """Process tractography and labels for a specific subject.
            - Encode labels to go from a node pair to a single integer (0,0)->0
            - Read streamlines from vtk file """
        print(f"Reading subject {subject_id}")
        subject_dir = os.path.join(self.input_dir, subject_id, "output")
        tractography_dir = os.path.join(subject_dir, "streamlines.vtk")
        streamlines, _ = read_tractography(tractography_dir, self.decay_factor)
        
        labels = np.zeros((len(streamlines), len(self.atlases)), dtype=int)
        for i, atlas in enumerate(self.atlases):
            encoded_labels_path = os.path.join(subject_dir, f"labels_{atlas}_{self.encoding}.txt")
            labels_path = os.path.join(subject_dir, f"labels_{atlas}.txt")
            encode_labels_txt(labels_path, encoded_labels_path, self.encoding, num_labels=self.num_labels[atlas])
            
            labels[:, i] = np.loadtxt(encoded_labels_path, dtype=int)

        subject_id_array = np.full(len(streamlines), subject_index, dtype=int)
        
        return streamlines, labels, subject_id_array

    def process_data(self):
        # Read subject IDs from file to determine which subjects to process
        self.read_subjects()
        
        global_subject_index = 0
        
        # Read data for all given subjects, do in chunks because of limited memory       
        for start in range(0, self.total_subjects, self.chunk_size):
            end = min(start + self.chunk_size, self.total_subjects)
            features_list = []
            labels_list = []
            subject_ids_list = []
            
            if self.chunk_size==1:
                streamlines, labels, subject_ids = self.process_subject(global_subject_index, self.subjects[global_subject_index])
                features_list.append(streamlines)
                labels_list.append(labels)
                subject_ids_list.append(subject_ids)
            else:
                # Run parallel jobs when enabled
                with ProcessPoolExecutor() as executor:
                    futures = [executor.submit(self.process_subject, global_subject_index + i, subject_id) 
                            for i, subject_id in enumerate(self.subjects[start:end])]
                    
                    for future in futures:
                        streamlines, labels, subject_ids = future.result()
                        if streamlines.shape[0] == labels.shape[0]:
                            features_list.append(streamlines)
                            labels_list.append(labels)
                            subject_ids_list.append(subject_ids)
                        else:
                            print(f"Mismatch in streamlines and labels for subject id {subject_ids[0]}")
            
            # Update the global subject index counter
            global_subject_index += (end - start)

            features = np.vstack(features_list)
            labels = np.vstack(labels_list)
            subject_ids = np.hstack(subject_ids_list)
            
            # Append to the output files
            self.save_data(features, labels, subject_ids)
    
    def save_data(self, features, labels, subject_ids):
        """Save data into train, validation, and test splits."""
        # Select labels that occur in less than the threshold across subjects
        rare_labels = {atlas: self.threshold_labels(labels[:, i], subject_ids, atlas) 
                for i, atlas in enumerate(self.atlases)}

        # Extract train, validation, and test data based on the subject splits
        train_subjects, val_subjects, test_subjects = self.split_subjects(subject_ids)
        data_splits = {
            'train': self.extract_data_by_subjects(subject_ids, train_subjects, features, labels),
            'val': self.extract_data_by_subjects(subject_ids, val_subjects, features, labels),
            'test': self.extract_data_by_subjects(subject_ids, test_subjects, features, labels)
        }
        
        self.save_data_to_pickle(data_splits, rare_labels)
        
        self.update_metadata(features, labels, subject_ids, {k: np.unique(v[2]) for k, v in data_splits.items()})

    def split_subjects(self, subject_ids):
        unique_subject_ids = np.unique(subject_ids)
        train_subjects, temp_subjects = train_test_split(
            unique_subject_ids, test_size=(1 - self.split_ratios[0]), random_state=42)
        val_subjects, test_subjects = train_test_split(
            temp_subjects, test_size=self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2]), random_state=42)
        return train_subjects, val_subjects, test_subjects
    
    def extract_data_by_subjects(self, subject_ids, selected_subjects, features, labels):
        """Extract data corresponding to specific subject IDs."""
        mask = np.isin(subject_ids.squeeze(), selected_subjects).squeeze()
        return features[mask], labels[mask], subject_ids[mask]
    
    def save_data_to_pickle(self, data_splits, rare_labels):
        
        for split, (X, y, subj) in data_splits.items():
            path = os.path.join(self.output_dir, f'{split}.pickle')
            data_dict = self.load_existing_data(path)
            
            data_dict = self.update_data_dict(data_dict, X, y, subj, rare_labels)
            
            with open(path, 'wb') as file:
                pickle.dump(data_dict, file)

    def load_existing_data(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
        return {}

    def update_data_dict(self, data_dict, X, y, subj, rare_labels):
        for i, atlas in enumerate(self.atlases):
            data_dict[f'label_{atlas}'] = self.concat_or_assign(data_dict.get(f'label_{atlas}'), y[:, i])
            data_dict[f'label_name_{atlas}'] = self.get_label_names(atlas)
            # Change all unknown to 0
            # data_dict[f'label_{atlas}_-1'] = [0 if isinstance(x, int) and 0 <= x < self.num_labels else x for x in data_dict[f'label_{atlas}_0']]
            
            # if self.threshold != 0: # Set rare labels to 0 and save this data as alternate labels with threshold
            #     thresholded_labels = np.where(np.isin(y[:, i], list(rare_labels[atlas])), 0, y[:, i])
            #     data_dict[f'label_{atlas}_{self.threshold}'] = self.concat_or_assign(
            #         data_dict.get(f'label_{atlas}_{self.threshold}'), thresholded_labels)
            #     data_dict[f'label_{atlas}_-{self.threshold}'] = [0 if isinstance(x, int) and 0 <= x < self.num_labels else x for x in data_dict[f'label_{atlas}_{self.threshold}']]
        
            # print(f'Labels with default settings for {atlas} atlas: {np.unique(data_dict[f"label_{atlas}_0"]).shape}')
            # print(f'Labels with grouping unknowns for {atlas} atlas: {np.unique(data_dict[f"label_{atlas}_-1"]).shape}')
            # print(f'Labels with thresholding for {atlas} atlas: {np.unique(data_dict[f"label_{atlas}_{self.threshold}"]).shape}')
            # print(f'Labels with both for {atlas} atlas: {np.unique(data_dict[f"label_{atlas}_-{self.threshold}"]).shape}')
        data_dict['feat'] = self.concat_or_assign(data_dict.get('feat'), X)
        data_dict['subject_id'] = self.concat_or_assign(data_dict.get('subject_id'), subj)
        
        return data_dict

    def concat_or_assign(self, existing_data, new_data):
        return np.concatenate([existing_data, new_data], axis=0) if existing_data is not None else new_data

    def get_label_names(self, atlas):
        if self.encoding == 'default':
            return [f"{i}_{j}" for i in range(self.num_labels[atlas]) for j in range(self.num_labels[atlas])]
        elif self.encoding == 'symmetric':
            return [f"{i}_{j}" for i in range(self.num_labels[atlas]) for j in range(i, self.num_labels[atlas])]
    
    def update_metadata(self, features, labels, subject_ids, subject_ids_split):
        """Update and save metadata based on processed data for multiple atlases."""
        
        metadata_path = os.path.join(self.output_dir, 'metadata.pickle')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as file:
                metadata = pickle.load(file)
        else:
            # Initialize metadata dictionaries for each atlas
            metadata = {
                'fibers_per_subject': {},
                'fibers_per_label': {atlas: {} for atlas in self.atlases},
                'labels_per_fiber_count': {atlas: {} for atlas in self.atlases},
                'split_subjects': {'train': [], 'val': [], 'test': []},
                'fibers_per_label_per_subject': {atlas: {} for atlas in self.atlases}
            }

        # Process and aggregate metadata for each subject and each atlas
        unique_subjects = np.unique(subject_ids)
        for subj in unique_subjects:
            # Collect the number of fibers per subject
            fibers_per_subject = np.sum(subject_ids == subj)
            metadata['fibers_per_subject'][subj] = fibers_per_subject
            
            # Process data for each atlas
            for i, atlas in enumerate(self.atlases):
                labels_for_subject = labels[subject_ids == subj][:, i]  # Get labels for this subject and atlas
                unique_labels, fibers_per_label = np.unique(labels_for_subject, return_counts=True)
                metadata['fibers_per_label_per_subject'][atlas][subj] = dict(zip(unique_labels, fibers_per_label))

                for label, count in zip(unique_labels, fibers_per_label):
                    if label in metadata['fibers_per_label'][atlas]:
                        metadata['fibers_per_label'][atlas][label] += count
                    else:
                        metadata['fibers_per_label'][atlas][label] = count

                # Collect the number of labels per fiber count
                for count in fibers_per_label:
                    if count in metadata['labels_per_fiber_count']:
                        metadata['labels_per_fiber_count'][atlas][count] += 1
                    else:
                        metadata['labels_per_fiber_count'][atlas][count] = 1

            # Track which subjects are in each split
            if subj in subject_ids_split['train']:
                split = 'train'
            elif subj in subject_ids_split['val']:
                split = 'val'
            elif subj in subject_ids_split['test']:
                split = 'test'

            metadata['split_subjects'][split].append(subj)
        
        # Calculate the number of detected and zero-fiber labels for each atlas
        for atlas in self.atlases:
            if self.encoding=='default':
                total_labels = self.num_labels[atlas]**2
            elif self.encoding=='symmetric':
                total_labels = ((self.num_labels[atlas]+1) * self.num_labels[atlas]) / 2
            detected_labels = len(metadata['fibers_per_label'][atlas])
            zero_fiber_labels = total_labels - detected_labels
            metadata['labels_per_fiber_count'][atlas][0] = zero_fiber_labels

        # Save the metadata
        with open(metadata_path, 'wb') as file:
            pickle.dump(metadata, file)
        
        print(f"Metadata successfully updated and saved to {metadata_path}")
    
    def threshold_labels(self, labels, subject_ids, atlas):
        """
        Set all labels to 0 if they are present in less than `threshold` percentage of subjects.

        Args:
        - labels: A numpy array of shape (n_samples,), where each entry is a label.
        - subject_ids: A numpy array of shape (n_samples,), where each entry corresponds to a subject ID.
        - threshold: The minimum percentage of subjects a label must be present in to avoid being set to 0 (default is 50%).

        Returns:
        - Adjusted labels where rare labels are set to 0.
        """
        # Get unique subject IDs
        unique_subjects = np.unique(subject_ids)

        # Initialize a dictionary to track how many subjects contain each label
        label_subject_count = {}

        # Iterate through the subjects and count how many subjects contain each label
        for subject in unique_subjects:
            subject_labels = np.unique(labels[subject_ids == subject])
            for label in subject_labels:
                if label not in label_subject_count:
                    label_subject_count[label] = 0
                label_subject_count[label] += 1

        # Determine the threshold count based on the number of subjects
        min_subject_count = len(unique_subjects) * (self.threshold/100)
        print(f"Labels must be present in at least {min_subject_count:.2f} subjects to avoid being set to 0 (threshold: {self.threshold}%).")

        # Create a set of labels that appear in less than the threshold number of subjects
        rare_labels = {label for label, count in label_subject_count.items() if count < min_subject_count}
        
        # Count how many streamlines will be affected
        streamlines_affected = np.sum(np.isin(labels, list(rare_labels)))
        
        print(f"Number of labels set to 0 (below threshold): {len(rare_labels)}")
        print(f"Number of streamlines affected (labels set to 0): {streamlines_affected}")
        print(f"Total streamlines: {len(labels)}")
        print(f"Percentage of streamlines affected: {(streamlines_affected / len(labels)) * 100:.2f}%")
        
        with open(os.path.join(self.output_dir, f'rare_labels_{self.threshold}%_{atlas}.txt'), 'w') as f:
            for rare_label in rare_labels:
                f.write(f"{rare_label}\n")
        
        return rare_labels

    
if __name__ == "__main__":
    # Parameters
    encoding='symmetric'
    threshold=50 # Percentage of subjects, labels should be present in
    decay_factor=0
    n_subjects=100
    # # Path
    base_dir="/media/volume/HCP_diffusion_MV/"
    output_dir = os.path.join(base_dir, f"TrainData_MRtrix_{n_subjects}_{encoding}_D{decay_factor}")
    TP = TractCloudPreprocessor(input_dir=os.path.join(base_dir, "data"),
                                output_dir=output_dir,
                                subjects_file=os.path.join(base_dir, f"subjects_tractography_output_{n_subjects}.txt"),
                                atlases=["aparc+aseg", "aparc.a2009s+aseg"],
                                encoding=encoding, threshold=threshold, decay_factor=decay_factor,
                                chunk_size=100, split_ratios=(0.7, 0.1, 0.2))
    
    # Make plots
    metadata = load_metadata(os.path.join(output_dir, 'metadata.pickle'))
    plot_fibers_per_subject(metadata, output_dir)
    for atlas in ["aparc+aseg", "aparc.a2009s+aseg"]:
        plot_fibers_per_label(metadata, output_dir, atlas=atlas, sort=False)
        plot_labels_per_fiber_count(metadata, output_dir, encoding, atlas=atlas)
        plot_labels_per_fiber_count(metadata, output_dir, encoding, atlas=atlas, xlim=50)
        plot_labels_per_fiber_count(metadata, output_dir, encoding, atlas=atlas, xlim=100)
        compute_fiber_stats(metadata, output_dir, atlas=atlas)
