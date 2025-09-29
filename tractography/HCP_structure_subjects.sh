#!/bin/bash

# Script to restructure the directories after downloading the HCP data
# in preparation for running MRtrix tractography

# Define the directory where the HCP data is downloaded
# HCP_download_dir=$1
# destination_dir=$2

# Remove all the .md5 files
rm ${HCP_download_dir}/*md5

# Loop over each zip file in the directory
for zip_file in "$HCP_download_dir"/*_3T_Diffusion_preproc.zip; do
    # Extract the subject ID from the zip file name
    filename=$(basename "$zip_file")
    subject_id="${filename%%_*}"

    # Create the destination directory
    destination_dir_dmri="${destination_dir}/${subject_id}/dMRI"
    mkdir -p "$destination_dir_dmri"

    # Unzip the files of interest into a temporary directory
    temp_directory=$(mktemp -d)
    unzip -j "$zip_file" "${subject_id}/T1w/Diffusion/bvals" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/T1w/Diffusion/bvecs" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/T1w/Diffusion/data.nii.gz" -d "$temp_directory"

    # Move the files to the destination directory
    mv "$temp_directory/bvals" "$destination_dir_dmri/"
    mv "$temp_directory/bvecs" "$destination_dir_dmri/"
    mv "$temp_directory/data.nii.gz" "$destination_dir_dmri/"

    # Remove the temporary directory
    rm -rf "$temp_directory"
    # rm -rf "$zip_file"
done

# Loop over each zip file in the directory
for zip_file in "$HCP_download_dir"/*_3T_Structural_preproc.zip; do
    # Extract the subject ID from the zip file name
    filename=$(basename "$zip_file")
    subject_id="${filename%%_*}"

    # Create the destination directory
    destination_dir_anat="${destination_dir}/${subject_id}/anat"
    mkdir -p "$destination_dir_anat"

    # Unzip the files of interest into a temporary directory
    temp_directory=$(mktemp -d)
    unzip -j "$zip_file" "${subject_id}/T1w/aparc+aseg.nii.gz" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/T1w/aparc.a2009s+aseg.nii.gz" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/T1w/T1w_acpc_dc_restore.nii.gz" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/T1w/T1w_acpc_dc_restore_brain.nii.gz" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/MNINonLinear/xfms/acpc_dc2standard.nii.gz" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/MNINonLinear/xfms/standard2acpc_dc.nii.gz" -d "$temp_directory"
    unzip -j "$zip_file" "${subject_id}/MNINonLinear/T1w_restore_brain.nii.gz" -d "$temp_directory"

    # Move the files to the destination directory
    mv "$temp_directory/aparc+aseg.nii.gz" "$destination_dir_anat/"
    mv "$temp_directory/aparc.a2009s+aseg.nii.gz" "$destination_dir_anat/"
    mv "$temp_directory/T1w_acpc_dc_restore.nii.gz" "$destination_dir_anat/"
    mv "$temp_directory/T1w_acpc_dc_restore_brain.nii.gz" "$destination_dir_anat/"
    mv "$temp_directory/acpc_dc2standard.nii.gz" "$destination_dir_anat/"
    mv "$temp_directory/standard2acpc_dc.nii.gz" "$destination_dir_anat/"
    mv "$temp_directory/T1w_restore_brain.nii.gz" "$destination_dir_anat/"

    # Remove the temporary directory
    rm -rf "$temp_directory"
    # rm -rf "$zip_file"
done

# Uncomment next line to delete the entire HCP download directory afterwards
# rm -rf "$HCP_download_dir"

