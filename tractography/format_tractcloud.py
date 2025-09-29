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
        tractography_dir = os.path.join(subject_dir, STREAMLINE_FILE)
        streamlines, _ = read_tractography(tractography_dir, self.decay_factor) 
        
        labels = np.zeros((len(streamlines), len(self.atlases)), dtype=int)
        print(labels.shape, subject_index)
        for i, atlas in enumerate(self.atlases):
            encoded_labels_path = os.path.join(subject_dir, f"labels_100K_{atlas}_{self.encoding}.txt")
            labels_path = os.path.join(subject_dir, f"labels_100K_{atlas}.txt")
            encode_labels_txt(labels_path, encoded_labels_path, self.encoding, num_labels=self.num_labels[atlas])
            
            labels[:, i] = np.loadtxt(encoded_labels_path, dtype=int)
        # Apply the filter to remove fibers with less than min_fibers labels
        # labels, streamlines = threshold(labels, streamlines, self.min_fibers)

        subject_id_array = np.full(len(streamlines), subject_index, dtype=int)
        # subject_id_array = np.full(labels.shape, subject_id, dtype=int) # use this if you want to work with indexes instead of the HCP ids
        
        return streamlines, labels, subject_id_array

    def extract_data_by_subjects(self, subject_ids, selected_subjects, features, labels):
        """Extract data corresponding to specific subject IDs."""
        # import pdb
        # pdb.set_trace()
        mask = np.isin(subject_ids.squeeze(), selected_subjects).squeeze()
        return features[mask], labels[mask], subject_ids[mask]

    def save_data(self, features, labels, subject_ids):
        """Save data into train, validation, and test splits."""
        
        # Select labels that occur in less than the threshold across subjects
        rare_labels = {atlas: threshold_labels(labels[:, i], subject_ids, threshold=self.threshold) 
                    for i, atlas in enumerate(self.atlases)}

        # Split the subject IDs into train, validation, and test sets
        unique_subject_ids = np.unique(subject_ids)
        self.train_subjects, temp_subjects = train_test_split(unique_subject_ids, test_size=(1 - self.split_ratios[0]), random_state=42)
        self.val_subjects, self.test_subjects = train_test_split(temp_subjects, test_size=self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2]), random_state=42)

        X_train, y_train, subj_train = self.extract_data_by_subjects(subject_ids, self.train_subjects, features, labels)
        X_val, y_val, subj_val = self.extract_data_by_subjects(subject_ids, self.val_subjects, features, labels)
        X_test, y_test, subj_test = self.extract_data_by_subjects(subject_ids, self.test_subjects, features, labels)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Paths for the split data
        paths = {
            'train': os.path.join(self.output_dir, 'train.pickle'),
            'val': os.path.join(self.output_dir, 'val.pickle'),
            'test': os.path.join(self.output_dir, 'test.pickle')
        }

        # Load existing data or create new dictionaries
        data_dict = {}
        for split in ['train', 'val', 'test']:
            if os.path.exists(paths[split]):
                with open(paths[split], 'rb') as file:
                    data_dict[split] = pickle.load(file)
            else:
                data_dict[split] = {'feat': [], 'subject_id': []}

        # Helper function to update the dictionaries with new data
        def update_dict(data, split_name, features, labels, subject_ids):
            for i, atlas in enumerate(self.atlases):
                if f'label_{atlas}' not in data:
                    data[f'label_{atlas}'] = labels[:, i]
                else:
                    data[f'label_{atlas}'] = np.concatenate([data[f'label_{atlas}'], labels[:, i]], axis=0)

                if self.threshold != 0:
                    label_key = f'label_{atlas}_{self.threshold}'
                    thresholded_labels = np.where(np.isin(labels[:, i], list(rare_labels[atlas])), 0, labels[:, i])
                    if label_key not in data:
                        data[label_key] = thresholded_labels
                    else:
                        data[label_key] = np.concatenate([data[label_key], thresholded_labels], axis=0)

            data['feat'] = np.concatenate([data['feat'], features], axis=0) if data['feat'] else features
            data['subject_id'] = np.concatenate([data['subject_id'], subject_ids], axis=0) if data['subject_id'] else subject_ids

        # Update the train, val, and test dictionaries
        update_dict(data_dict['train'], 'train', X_train, y_train, subj_train)
        update_dict(data_dict['val'], 'val', X_val, y_val, subj_val)
        update_dict(data_dict['test'], 'test', X_test, y_test, subj_test)

        # Save the updated dictionaries to files
        for split in ['train', 'val', 'test']:
            with open(paths[split], 'wb') as file:
                pickle.dump(data_dict[split], file)

        # Store subject splits
        subject_ids_split = {
            "train": np.unique(subj_train),
            "val": np.unique(subj_val),
            "test": np.unique(subj_test)
        }

        # Update metadata
        self.update_metadata(features, labels, subject_ids, subject_ids_split)

    def save_data(self, features, labels, subject_ids):
        """Save data into train, validation, and test splits."""
        
        # Select labels that occur only in a certain portion of the population
        rare_labels={}
        for i, atlas in enumerate(self.atlases):
            rare_labels[atlas] = threshold_labels(labels[:, i], subject_ids, threshold=self.threshold)   
        
        # Extract train, validation, and test data based on the subject splits
        # Split the subject IDs into train, temp (val + test)
        unique_subject_ids = np.unique(subject_ids)
        self.train_subjects, temp_subjects = train_test_split(
            unique_subject_ids, test_size=(1 - self.split_ratios[0]), random_state=42)
        self.val_subjects, self.test_subjects = train_test_split(
            temp_subjects, test_size=self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2]), random_state=42)
        
        X_train, y_train, subj_train = self.extract_data_by_subjects(subject_ids, self.train_subjects, features, labels)
        X_val, y_val, subj_val = self.extract_data_by_subjects(subject_ids, self.val_subjects, features, labels)
        X_test, y_test, subj_test = self.extract_data_by_subjects(subject_ids, self.test_subjects, features, labels)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        train_path = os.path.join(self.output_dir, 'train.pickle')
        val_path = os.path.join(self.output_dir, 'val.pickle')
        test_path = os.path.join(self.output_dir, 'test.pickle')

        # Initialize dictionaries for storing data
        print("Saving data to pickle files")
        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            with open(train_path, 'rb') as file:
                train_dict = pickle.load(file)
            with open(val_path, 'rb') as file:
                val_dict = pickle.load(file)
            with open(test_path, 'rb') as file:
                test_dict = pickle.load(file)

            # Concatenate the new data with the existing data
            for i, atlas in enumerate(self.atlases):
                train_dict[f'label_{atlas}'] = np.concatenate([train_dict[f'label_{atlas}'], y_train[:, i]], axis=0)
                val_dict[f'label_{atlas}'] = np.concatenate([val_dict[f'label_{atlas}'], y_val[:, i]], axis=0)
                test_dict[f'label_{atlas}'] = np.concatenate([test_dict[f'label_{atlas}'], y_test[:, i]], axis=0)
                if self.threshold!=0: # Set rare labels to 0
                    
                    
                    train_dict[f'label_{atlas}_{self.threshold}'] = np.concatenate([train_dict[f'label_{atlas}_{self.threshold}'], np.where(np.isin(y_train[:, i], list(rare_labels[atlas])), 0, y_train[:, i])], axis=0)
                    val_dict[f'label_{atlas}_{self.threshold}'] = np.concatenate([val_dict[f'label_{atlas}_{self.threshold}'], np.where(np.isin(y_val[:, i], list(rare_labels[atlas])), 0, y_val[:, i])], axis=0)
                    test_dict[f'label_{atlas}_{self.threshold}'] = np.concatenate([test_dict[f'label_{atlas}_{self.threshold}'], np.where(np.isin(y_test[:, i], list(rare_labels[atlas])), 0, y_test[:, i])], axis=0)
                
            train_dict['feat'] = np.concatenate([train_dict['feat'], X_train], axis=0)
            train_dict['subject_id'] = np.concatenate([train_dict['subject_id'], subj_train], axis=0)
            val_dict['feat'] = np.concatenate([val_dict['feat'], X_val], axis=0)
            val_dict['subject_id'] = np.concatenate([val_dict['subject_id'], subj_val], axis=0)
            test_dict['feat'] = np.concatenate([test_dict['feat'], X_test], axis=0)
            test_dict['subject_id'] = np.concatenate([test_dict['subject_id'], subj_test], axis=0)

        else:
            # Initialize the dictionaries if the files don't exist
            train_dict = {}
            val_dict = {}
            test_dict = {}

            for i, atlas in enumerate(self.atlases):
                train_dict[f'label_{atlas}'] = y_train[:, i]
                val_dict[f'label_{atlas}'] = y_val[:, i]
                test_dict[f'label_{atlas}'] = y_test[:, i]
                if self.threshold!=0:
                    train_dict[f'label_{atlas}_{self.threshold}'] = np.where(np.isin(y_train[:, i], list(rare_labels[atlas])), 0, y_train[:, i])
                    val_dict[f'label_{atlas}_{self.threshold}'] =  np.where(np.isin(y_val[:, i], list(rare_labels[atlas])), 0, y_val[:, i])
                    test_dict[f'label_{atlas}_{self.threshold}'] =  np.where(np.isin(y_test[:, i], list(rare_labels[atlas])), 0, y_test[:, i])
                    
            train_dict['feat'] = X_train
            train_dict['subject_id'] = subj_train
            val_dict['feat'] = X_val
            val_dict['subject_id'] = subj_val
            test_dict['feat'] = X_test
            test_dict['subject_id'] = subj_test
            
            # Determine the number of labels based on encoding type
            for i, atlas in enumerate(self.atlases):
                if self.encoding == 'default':
                    label_names = [f"{i}_{j}" for i in range(self.num_labels[atlas]) for j in range(self.num_labels[atlas])]
                elif self.encoding == 'symmetric':
                    label_names = [f"{i}_{j}" for i in range(self.num_labels[atlas]) for j in range(i, self.num_labels[atlas])]
                train_dict[f'label_name_{atlas}'] = label_names
                val_dict[f'label_name_{atlas}'] = label_names
                test_dict[f'label_name_{atlas}'] = label_names
        
        # Save the updated data to pickle files
        with open(train_path, 'wb') as file:
            pickle.dump(train_dict, file)
        with open(val_path, 'wb') as file:
            pickle.dump(val_dict, file)
        with open(test_path, 'wb') as file:
            pickle.dump(test_dict, file)

        # Store unique subject IDs for each split
        subject_ids_split = {
            "train": np.unique(subj_train),
            "val": np.unique(subj_val),
            "test": np.unique(subj_test)
        }

        self.update_metadata(features, labels, subject_ids, subject_ids_split)

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
            # import pdb
            # pdb.set_trace()
            features = np.vstack(features_list)
            labels = np.vstack(labels_list)
            subject_ids = np.hstack(subject_ids_list)
            
            # Append to the output files
            self.save_data(features, labels, subject_ids)
    
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
    
def threshold_labels(labels, subject_ids, threshold):
    """
    Set all labels to 0 if they are present in less than threshold percentage of subjects.

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
    min_subject_count = len(unique_subjects) * (threshold/100)
    # print(f"Labels must be present in at least {min_subject_count:.2f} subjects to avoid being set to 0 (threshold: {threshold}%).")

    # Create a set of labels that appear in less than the threshold number of subjects
    rare_labels = {label for label, count in label_subject_count.items() if count < min_subject_count}
    
    # Count how many streamlines will be affected
    streamlines_affected = np.sum(np.isin(labels, list(rare_labels)))
    
    # print(f"Number of labels set to 0 (below threshold): {len(rare_labels)}")
    # print(f"Number of streamlines affected (labels set to 0): {streamlines_affected}")
    # print(f"Total streamlines: {len(labels)}")
    # print(f"Percentage of streamlines affected: {(streamlines_affected / len(labels)) * 100:.2f}%")
    
    return rare_labels

    
if __name__ == "__main__":
    # Parameters
    STREAMLINE_FILE="streamlines_100K_MNI.vtk"
    encoding='symmetric'
    threshold=0 # Percentage of subjects labels should be present in
    decay_factor=0
    # Path
    base_dir="/media/volume/MV_HCP"
    output_dir = os.path.join(base_dir, f"TrainData_MRtrix_1000_MNI_100K_2")
    TP = TractCloudPreprocessor(input_dir=os.path.join(base_dir, "HCP_MRtrix"),
                                output_dir=output_dir,
                                subjects_file="/media/volume/MV_HCP/subjects_tractography_output_1000.txt",
                                atlases=["aparc+aseg", "aparc.a2009s+aseg"],
                                encoding=encoding, threshold=threshold, decay_factor=decay_factor,
                                chunk_size=100, split_ratios=(0.7, 0.1, 0.2))
    
    # Make plots
    # metadata = load_metadata(os.path.join(output_dir, 'metadata.pickle'))
    # plot_fibers_per_subject(metadata, output_dir)
    # for atlas in ["aparc+aseg", "aparc.a2009s+aseg"]:
    #     plot_fibers_per_label(metadata, output_dir, atlas=atlas, sort=False)
    #     plot_labels_per_fiber_count(metadata, output_dir, encoding, atlas=atlas)
    #     plot_labels_per_fiber_count(metadata, output_dir, encoding, atlas=atlas, xlim=50)
    #     plot_labels_per_fiber_count(metadata, output_dir, encoding, atlas=atlas, xlim=100)
    #     compute_fiber_stats(metadata, output_dir, atlas=atlas)
        
        