"""Prepare DeepMultiConnectome training data from tractography outputs."""

import os
import numpy as np
import whitematteranalysis as wma
import pickle
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import sys

sys.path.append('..')
from utils.tract_feat import CustomFiberArray
from utils.label_encoding import encode_labels_file


# ==============================================================================
# Tractography Processing
# ==============================================================================

def read_tractography(tractography_path, decay_factor=0):
    """Read tractography and convert to point cloud representation."""
    pd_tractography = wma.io.read_polydata(tractography_path)
    fiber_array = CustomFiberArray()
    fiber_array.convert_from_polydata(
        pd_tractography,
        points_per_fiber=15,
        distribution='exponential',
        decay_factor=decay_factor
    )
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    return feat


def threshold_labels(labels, subject_ids, threshold):
    """Identify labels present in less than threshold percent of subjects."""
    unique_subjects = np.unique(subject_ids)
    label_subject_count = {}
    
    for subject in unique_subjects:
        subject_labels = np.unique(labels[subject_ids == subject])
        for label in subject_labels:
            label_subject_count[label] = label_subject_count.get(label, 0) + 1
    
    min_subject_count = len(unique_subjects) * (threshold / 100)
    rare_labels = {label for label, count in label_subject_count.items() 
                   if count < min_subject_count}
    
    return rare_labels


# ==============================================================================
# Main Preprocessing Class
# ==============================================================================

class DeepMultiConnectomePreprocessor:
    """Preprocess HCP tractography data for DeepMultiConnectome training."""
    
    def __init__(self, input_dir, output_dir, subjects_file, atlases,
                 encoding='symmetric', threshold=0, decay_factor=0,
                 split_ratios=(0.7, 0.1, 0.2), chunk_size=1, streamlines_tag="10M"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects_file = subjects_file
        self.atlases = atlases
        self.encoding = encoding
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.split_ratios = split_ratios
        self.chunk_size = chunk_size
        self.streamlines_tag = streamlines_tag
        
        self.num_labels = {"aparc+aseg": 85, "aparc.a2009s+aseg": 165}
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.process_data()
    
    def read_subjects(self):
        """Load subject IDs from file."""
        with open(self.subjects_file, 'r') as f:
            self.subjects = f.read().splitlines()
        self.total_subjects = len(self.subjects)
        print(f"Total subjects to process: {self.total_subjects}")
    
    def process_subject(self, subject_index, subject_id):
        """Process a single subject and return features and labels."""
        print(f"Processing subject {subject_id}")
        subject_dir = os.path.join(self.input_dir, subject_id, "output")
        tractography_path = os.path.join(
            subject_dir, f"streamlines_{self.streamlines_tag}.vtk"
        )

        if not os.path.exists(tractography_path):
            raise FileNotFoundError(f"Missing tractography: {tractography_path}")
        
        streamlines = read_tractography(tractography_path, self.decay_factor)
        labels = np.zeros((len(streamlines), len(self.atlases)), dtype=int)
        
        for i, atlas in enumerate(self.atlases):
            labels_path = os.path.join(
                subject_dir, f"labels_{self.streamlines_tag}_{atlas}.txt"
            )
            encoded_labels_path = os.path.join(
                subject_dir, f"labels_{self.streamlines_tag}_{atlas}_{self.encoding}.txt"
            )

            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Missing labels: {labels_path}")
            
            # Encode labels if not already done
            if not os.path.exists(encoded_labels_path):
                encode_labels_file(labels_path, encoded_labels_path, self.encoding, 
                                  num_labels=self.num_labels[atlas])
            
            labels[:, i] = np.loadtxt(encoded_labels_path, dtype=int)
        
        subject_id_array = np.full(len(streamlines), subject_index, dtype=int)
        return streamlines, labels, subject_id_array
    
    def extract_data_by_subjects(self, subject_ids, selected_subjects, features, labels):
        """Extract data for specific subjects."""
        mask = np.isin(subject_ids.squeeze(), selected_subjects)
        return features[mask], labels[mask], subject_ids[mask]
    
    def save_data(self, features, labels, subject_ids):
        """Save data with train/val/test splits."""
        if self.threshold > 0:
            rare_labels = {
                atlas: threshold_labels(labels[:, i], subject_ids, self.threshold)
                for i, atlas in enumerate(self.atlases)
            }
        else:
            rare_labels = {atlas: set() for atlas in self.atlases}
        
        # Split subjects
        unique_subject_ids = np.unique(subject_ids)
        train_subj, temp_subj = train_test_split(unique_subject_ids, 
                                                  test_size=(1 - self.split_ratios[0]), 
                                                  random_state=42)
        val_subj, test_subj = train_test_split(temp_subj, 
                                               test_size=self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2]), 
                                               random_state=42)
        
        # Extract split data
        X_train, y_train, subj_train = self.extract_data_by_subjects(subject_ids, train_subj, features, labels)
        X_val, y_val, subj_val = self.extract_data_by_subjects(subject_ids, val_subj, features, labels)
        X_test, y_test, subj_test = self.extract_data_by_subjects(subject_ids, test_subj, features, labels)
        
        # Load or create data dictionaries
        train_path = os.path.join(self.output_dir, 'train.pickle')
        val_path = os.path.join(self.output_dir, 'val.pickle')
        test_path = os.path.join(self.output_dir, 'test.pickle')
        
        if os.path.exists(train_path):
            # Append to existing data
            with open(train_path, 'rb') as f:
                train_dict = pickle.load(f)
            with open(val_path, 'rb') as f:
                val_dict = pickle.load(f)
            with open(test_path, 'rb') as f:
                test_dict = pickle.load(f)
            
            # Concatenate features and labels
            for i, atlas in enumerate(self.atlases):
                train_dict[f'label_{atlas}'] = np.concatenate([train_dict[f'label_{atlas}'], y_train[:, i]])
                val_dict[f'label_{atlas}'] = np.concatenate([val_dict[f'label_{atlas}'], y_val[:, i]])
                test_dict[f'label_{atlas}'] = np.concatenate([test_dict[f'label_{atlas}'], y_test[:, i]])
                
                if self.threshold != 0:
                    train_dict[f'label_{atlas}_{self.threshold}'] = np.concatenate([
                        train_dict[f'label_{atlas}_{self.threshold}'],
                        np.where(np.isin(y_train[:, i], list(rare_labels[atlas])), 0, y_train[:, i])
                    ])
                    val_dict[f'label_{atlas}_{self.threshold}'] = np.concatenate([
                        val_dict[f'label_{atlas}_{self.threshold}'],
                        np.where(np.isin(y_val[:, i], list(rare_labels[atlas])), 0, y_val[:, i])
                    ])
                    test_dict[f'label_{atlas}_{self.threshold}'] = np.concatenate([
                        test_dict[f'label_{atlas}_{self.threshold}'],
                        np.where(np.isin(y_test[:, i], list(rare_labels[atlas])), 0, y_test[:, i])
                    ])
            
            train_dict['feat'] = np.concatenate([train_dict['feat'], X_train])
            train_dict['subject_id'] = np.concatenate([train_dict['subject_id'], subj_train])
            val_dict['feat'] = np.concatenate([val_dict['feat'], X_val])
            val_dict['subject_id'] = np.concatenate([val_dict['subject_id'], subj_val])
            test_dict['feat'] = np.concatenate([test_dict['feat'], X_test])
            test_dict['subject_id'] = np.concatenate([test_dict['subject_id'], subj_test])
        else:
            # Create new dictionaries
            train_dict = {'feat': X_train, 'subject_id': subj_train}
            val_dict = {'feat': X_val, 'subject_id': subj_val}
            test_dict = {'feat': X_test, 'subject_id': subj_test}
            
            for i, atlas in enumerate(self.atlases):
                train_dict[f'label_{atlas}'] = y_train[:, i]
                val_dict[f'label_{atlas}'] = y_val[:, i]
                test_dict[f'label_{atlas}'] = y_test[:, i]
                
                if self.threshold != 0:
                    train_dict[f'label_{atlas}_{self.threshold}'] = np.where(
                        np.isin(y_train[:, i], list(rare_labels[atlas])), 0, y_train[:, i])
                    val_dict[f'label_{atlas}_{self.threshold}'] = np.where(
                        np.isin(y_val[:, i], list(rare_labels[atlas])), 0, y_val[:, i])
                    test_dict[f'label_{atlas}_{self.threshold}'] = np.where(
                        np.isin(y_test[:, i], list(rare_labels[atlas])), 0, y_test[:, i])
                
                # Generate label names
                if self.encoding == 'default':
                    label_names = [f"{i}_{j}" for i in range(self.num_labels[atlas]) 
                                  for j in range(self.num_labels[atlas])]
                elif self.encoding == 'symmetric':
                    label_names = [f"{i}_{j}" for i in range(self.num_labels[atlas]) 
                                  for j in range(i, self.num_labels[atlas])]
                
                train_dict[f'label_name_{atlas}'] = label_names
                val_dict[f'label_name_{atlas}'] = label_names
                test_dict[f'label_name_{atlas}'] = label_names
        
        # Save pickle files
        with open(train_path, 'wb') as f:
            pickle.dump(train_dict, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_dict, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_dict, f)
        
        print(f"Saved data to {self.output_dir}")
        print(f"  Train: {len(train_dict['feat'])} streamlines from {len(np.unique(subj_train))} subjects")
        print(f"  Val: {len(val_dict['feat'])} streamlines from {len(np.unique(subj_val))} subjects")
        print(f"  Test: {len(test_dict['feat'])} streamlines from {len(np.unique(subj_test))} subjects")
    
    def process_data(self):
        """Main processing loop."""
        self.read_subjects()
        
        global_subject_index = 0
        
        for start in range(0, self.total_subjects, self.chunk_size):
            end = min(start + self.chunk_size, self.total_subjects)
            features_list = []
            labels_list = []
            subject_ids_list = []
            
            if self.chunk_size == 1:
                streamlines, labels, subject_ids = self.process_subject(
                    global_subject_index, self.subjects[global_subject_index])
                features_list.append(streamlines)
                labels_list.append(labels)
                subject_ids_list.append(subject_ids)
            else:
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
                            print(f"Skipping subject {subject_ids[0]}: shape mismatch")
            
            global_subject_index += (end - start)
            
            # Combine chunk data
            features = np.vstack(features_list)
            labels = np.vstack(labels_list)
            subject_ids = np.hstack(subject_ids_list)
            
            # Save chunk
            self.save_data(features, labels, subject_ids)


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    base_dir = "/path/to/HCP"
    subjects_file = "/path/to/subjects.txt"

    output_dir = os.path.join(base_dir, "TrainData_DeepMultiConnectome")

    DeepMultiConnectomePreprocessor(
        input_dir=os.path.join(base_dir, "HCP_MRtrix"),
        output_dir=output_dir,
        subjects_file=subjects_file,
        atlases=["aparc+aseg", "aparc.a2009s+aseg"],
        encoding='symmetric',
        threshold=0,
        decay_factor=0,
        split_ratios=(0.7, 0.1, 0.2),
        chunk_size=100,
        streamlines_tag="10M"
    )

    print(f"\nData preparation complete. Output saved to: {output_dir}")
