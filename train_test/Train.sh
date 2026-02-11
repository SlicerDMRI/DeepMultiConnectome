#!/bin/bash
# Training params
model_name="pointnet"                   # model: dgcnn or pointnet
epoch=150                               # epochs
batch_size=1024                         # batch size
lr=1e-3                                 # learning rate

# Data
# input_data should match the folder produced by data/prepare_training_data.py
# Example: TrainData_${input_data} -> ../TrainData_MRtrix_1000_MNI_100K
input_data="MRtrix_1000_MNI_100K"       # training data tag
num_f_brain=10000                       # number of streamlines per brain
num_p_fiber=15                          # number of points per streamline
atlas="aparc+aseg,aparc.a2009s+aseg"    # aparc+aseg, aparc.a2009s+aseg, or both

# Paths
# Adjust base paths to your environment.
train_data_base=".."                    # folder containing TrainData_${input_data}
weights_base=".."                       # folder to store ModelWeights

param_folder=bs${batch_size}_nf${num_f_brain}_epoch${epoch}_lr${lr}_${atlas}
out_path=${weights_base}/ModelWeights/Data${input_data}_${model_name}/${param_folder}
input_path=${train_data_base}/TrainData_${input_data}

######### Train/Validation/Test #########
python train.py ---connectome --atlas ${atlas} --num_fiber_per_brain ${num_f_brain} --num_point_per_fiber ${num_p_fiber} --input_path ${input_path} --epoch ${epoch} --out_path_base ${out_path} --model_name $model_name --train_batch_size ${batch_size} --val_batch_size ${batch_size} --test_batch_size ${batch_size} --lr ${lr}
python test.py --connectome --atlas ${atlas} --out_path_base ${out_path} --input_path ${input_path}

