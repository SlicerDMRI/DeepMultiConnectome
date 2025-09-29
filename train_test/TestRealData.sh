#!/bin/bash
# Training params
model_name="pointnet"              # model
epoch=150                        # epoch
batch_size=1024                 # batch size
lr=1e-3                         # learning rate
threshold=0

# Data
input_data="MRtrix_1000_MNI_100K"         # training data, 800 clusters + 800 outliers
num_f_brain=10000              # the number of streamlines in a brain
num_p_fiber=15                  # the number of points on a streamline
atlas="aparc+aseg,aparc.a2009s+aseg"

# Local-global representation
k="0"   # local, neighbor streamlines
k_global="0"   # global, randomly selected streamlines in the whole-brain
k_ds_rate=0.1  # downsample the tractography when calculating neighbor streamlines

local_global_rep_folder=k${k}_kg${k_global}_bs${batch_size}_nf${num_f_brain}_epoch${epoch}_lr${lr}_THR${threshold}_${atlas}
weight_path_base=../ModelWeights/Data${input_data}_${model_name}/${local_global_rep_folder}/
num_classes='3655,13695'

# Prompt for inference mode
echo "Choose inference mode:"
echo "1 - Single subject"
echo "2 - Test set"
echo "3 - Test-retest set"
read -p "Enter the mode number (1/2/3): " mode

if [[ $mode -eq 1 ]]; then
    # Single subject inference
    read -p "Enter subject ID: " subject_idx
    tractography_path=/media/volume/MV_HCP/HCP_MRtrix/${subject_idx}/output/streamlines.vtk
    out_path=/media/volume/MV_HCP/HCP_MRtrix/${subject_idx}/TractCloud/
    python test_realdata.py --atlas ${atlas} \
        --weight_path_base ${weight_path_base} --tractography_path ${tractography_path} --out_path ${out_path} \
        --test_realdata_batch_size ${batch_size} --k_ds_rate ${k_ds_rate}

elif [[ $mode -eq 2 ]]; then
    # Test set inference
    subject_file="/media/volume/MV_HCP/subjects_tractography_output_1000_test.txt"
    data_path_base="/media/volume/MV_HCP/HCP_MRtrix"
    streamlines="streamlines_10M_MNI.vtk"

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

elif [[ $mode -eq 3 ]]; then
    # Test-retest set inference
    subject_file="/media/volume/MV_HCP/subjects_tractography_output_TRT.txt"
    streamlines="streamlines_10M_MNI.vtk"

    while IFS= read -r subject_idx; do
        # TEST
        data_path_base="/media/volume/MV_HCP/HCP_MRtrix_test"
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

        # RETEST
        data_path_base="/media/volume/MV_HCP/HCP_MRtrix_retest"
        tractography_path=${data_path_base}/${subject_idx}/output/${streamlines}
        out_path=${data_path_base}/${subject_idx}/TractCloud/
        prediction_file="${out_path}/predictions_aparc+aseg.txt"

        if [[ ! -f "$prediction_file" ]]; then
            python test_realdata.py --atlas ${atlas} \
                --weight_path_base ${weight_path_base} --tractography_path ${tractography_path} --out_path ${out_path} \
                --test_realdata_batch_size ${batch_size} --k_ds_rate ${k_ds_rate}
        else
            echo "${subject_idx} retest prediction already exists."
        fi
    done < "$subject_file"
else
    echo "Invalid mode selected. Exiting."
    exit 1
fi
