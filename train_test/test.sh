#!/bin/bash
# Paths (adjust to your environment)
weights_base=".."                       # folder containing ModelWeights
data_path_base="/path/to/HCP_MRtrix"    # base folder for subject data
subject_file_test="/path/to/subjects_test.txt" # text file containing test subject IDs, one per line

# Training params
model_name="pointnet"              # model
epoch=150                        # epoch
batch_size=1024                 # batch size
lr=1e-3                         # learning rate
threshold=0

# Data
# input_data should match the folder produced by data/prepare_training_data.py
input_data="MRtrix_1000_MNI_100K"         # training data tag
num_f_brain=10000              # number of streamlines per brain
num_p_fiber=15                  # number of points per streamline
atlas="aparc+aseg,aparc.a2009s+aseg"

# Local-global representation
k="0"   # local, neighbor streamlines
k_global="0"   # global, randomly selected streamlines in the whole-brain
k_ds_rate=0.1  # downsample the tractography when calculating neighbor streamlines

num_classes='3655,13695'

# ============================================================================
# CHOOSE MODEL SOURCE
# ============================================================================
echo "================================"
echo "DeepMultiConnectome Inference"
echo "================================"
echo ""
echo "Choose model source:"
echo "1 - Default trained model (trained_model.pth)"
echo "2 - Parameter-based model path"
read -p "Enter the model source (1/2) [default: 1]: " model_choice
model_choice=${model_choice:-1}

if [[ $model_choice -eq 1 ]]; then
    # Use default trained model in current directory
    weight_path_base="trained_model/"
    echo "Using pre-trained default model"
elif [[ $model_choice -eq 2 ]]; then
    # Use parameter-based path
    local_global_rep_folder=k${k}_kg${k_global}_bs${batch_size}_nf${num_f_brain}_epoch${epoch}_lr${lr}_THR${threshold}_${atlas}
    weight_path_base=${weights_base}/ModelWeights/Data${input_data}_${model_name}/${local_global_rep_folder}/
    echo "Using parameter-based model path: ${weight_path_base}"
else
    echo "Invalid model source selected. Exiting."
    exit 1
fi

echo ""
# ============================================================================
# CHOOSE INFERENCE MODE
# ============================================================================
echo "Choose inference mode:"
echo "1 - Single subject"
echo "2 - Test set"
read -p "Enter the mode number (1/2): " mode

if [[ $mode -eq 1 ]]; then
    # Single subject inference
    read -p "Enter subject ID: " subject_idx
    tractography_path=${data_path_base}/${subject_idx}/output/streamlines.vtk
    out_path=${data_path_base}/${subject_idx}/TractCloud/
    python test_realdata.py --atlas ${atlas} \
        --weight_path_base ${weight_path_base} --tractography_path ${tractography_path} --out_path ${out_path} \
        --test_realdata_batch_size ${batch_size} --k_ds_rate ${k_ds_rate}

elif [[ $mode -eq 2 ]]; then
    # Test set inference
    subject_file=${subject_file_test}
    streamlines="streamlines.vtk"

    while IFS= read -r subject_idx; do
        tractography_path=${data_path_base}/${subject_idx}/output/${streamlines}
        out_path=${data_path_base}/${subject_idx}/TractCloud/
        prediction_file="${out_path}/predictions_aparc+aseg.txt"

        if [[ ! -f "$prediction_file" ]]; then
            python test_realdata.py --atlas ${atlas} \
                --weight_path_base ${weight_path_base} --tractography_path ${tractography_path} --out_path ${out_path} \
                --test_realdata_batch_size ${batch_size} --k_ds_rate ${k_ds_rate}
        else
            echo "${subject_idx} test prediction already exists."
        fi
    done < "$subject_file"

else
    echo "Invalid mode selected. Exiting."
    exit 1
fi
