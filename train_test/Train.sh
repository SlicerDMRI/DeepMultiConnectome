#!/bin/bash
# Training params
model_name="pointnet"                   # model dgcnn or pointnet
epoch=150                               # epoch
batch_size=1024                         # batch size
lr=1e-3                                 # learning rate

# Data
input_data="MRtrix_1000_MNI_100K"       # training data
num_f_brain=10000                       # the number of streamlines in a brain
num_p_fiber=15                          # the number of points on a streamline
atlas="aparc+aseg,aparc.a2009s+aseg"    # names of atlases: aparc+aseg or aparc.a2009s+aseg or aparc+aseg,aparc.a2009s+aseg

# Paths
param_folder=bs${batch_size}_nf${num_f_brain}_epoch${epoch}_lr${lr}_${atlas}
out_path=../ModelWeights/Data${input_data}_${model_name}/${param_folder}
input_path=../TrainData_${input_data}

######### Train/Validation/Test #########
python train.py ---connectome --atlas ${atlas} --num_fiber_per_brain ${num_f_brain} --num_point_per_fiber ${num_p_fiber} --input_path ${input_path} --epoch ${epoch} --out_path_base ${out_path} --model_name $model_name --train_batch_size ${batch_size} --val_batch_size ${batch_size} --test_batch_size ${batch_size}  --lr ${lr}
python test.py --connectome --atlas ${atlas} --out_path_base ${out_path} --input_path ${input_path}

