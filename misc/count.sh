#!/bin/bash

# data_dir=$1
data_dir=/media/volume/MV_HCP/HCP_MRtrix
output_file="subjects_tractography_output.txt"

# Clear the output file if it exists
> "$output_file"

total_subjects=0
subjects_with_output=0
subjects_with_anat=0
subjects_with_dmri=0
subjects_deleted=0

# Counters for specific files
streamlines_vtk_count=0
streamlines_vtk_count_MNI=0
labels_txt1_count=0
labels_txt2_count=0
connectome_matrix_png_count=0
labels_encoded_txt_count=0

# Function to check for empty output folders
check_empty_output_folders() {
    for subject_dir in ${data_dir}/* ; do
        output_dir="${subject_dir}/output"
        
        # if [ -d "$output_dir" ] && [ -z "$(ls -A "$output_dir")" ]; then
            # echo "Empty output folder: $output_dir"
        rm -rf "$output_dir"
        # fi
    done
}

# Loop through each subject directory
for subject_dir in ${data_dir}/* ; do
    ((total_subjects++))
    
    anat_dir="${subject_dir}/anat"
    dmri_dir="${subject_dir}/dMRI"
    output_dir="${subject_dir}/output"
    
    if [ -d "${output_dir}" ]; then
        ((subjects_with_output++))

        # Check for specific files in the output directory
        [ -f "${output_dir}/streamlines_100K.vtk" ] && ((streamlines_vtk_count++))
        [ -f "${output_dir}/streamlines_100K_MNI.vtk" ] && ((streamlines_vtk_count_MNI++))
        [ -f "${output_dir}/labels_100K_aparc+aseg.txt" ] && ((labels_txt1_count++))
        [ -f "${output_dir}/labels_100K_aparc.a2009s+aseg.txt" ] && ((labels_txt2_count++))
        [ -f "${output_dir}/connectome_matrix_aparc+aseg.csv" ] && ((connectome_matrix_png_count++))
        [ -f "${output_dir}/labels_encoded.txt" ] && ((labels_encoded_txt_count++))

        if [ -f "${output_dir}/streamlines_100K.vtk" ] && [ -f "${output_dir}/labels_100K_aparc+aseg.txt" ]; then
            subject_id=$(basename "$subject_dir")
            echo "$subject_id" >> "$output_file"

            # Encode the labels from labels.txt to labels_encoded.txt
            # python label_encoder.py encode default "${output_dir}/labels.txt" "${output_dir}/labels_encoded_default.txt"
            # python label_encoder.py encode symmetric "${output_dir}/labels.txt" "${output_dir}/labels_encoded_symmetric.txt"
        fi

        if ! test -f "${output_dir}/streamlines_100K.vtk"; then
            echo "${output_dir}"
            # rm -rf ${output_dir}


        # # else
        # #     rm ${anat_dir}/*.mif
        # #     rm ${dmri_dir}/*.mif
        fi
    # else
        # mv "${dmri_dir}/bvals" "${subject_dir}/"
        # mv "${dmri_dir}/bvecs" "${subject_dir}/"
        # mv "${dmri_dir}/data.nii.gz" "${subject_dir}/"
        # rm -rf ${dmri_dir}

        # mkdir ${dmri_dir}
        # mv "${subject_dir}/bvals" "${dmri_dir}/"
        # mv "${subject_dir}/bvecs" "${dmri_dir}/"
        # mv "${subject_dir}/data.nii.gz" "${dmri_dir}/"
    #     echo "no output dir for ${subject_dir}"
    #     # rm ${anat_dir}/*.mif
    #     # rm ${dmri_dir}/*.mif
    #     # rm ${dmri_dir}/*.txt
    fi
    
    if [ -f "${anat_dir}/T1w_acpc_dc_restore.nii.gz" ]; then
        ((subjects_with_anat++))
    fi
    
    if [ -f "${dmri_dir}/data.nii.gz" ]; then
        ((subjects_with_dmri++))
    else
        # Optionally delete the subject directory if it does not have the dMRI folder
        # rm -rf "${subject_dir}"
        ((subjects_deleted++))
    fi
done

# Call the function to check for empty output folders
# check_empty_output_folders
echo " "
echo "Total subjects: $total_subjects"
echo "Subjects with anat: $subjects_with_anat"
echo "Subjects with dMRI: $subjects_with_dmri"
echo "Subjects with output folder: $subjects_with_output"
echo " "
echo "Subjects with streamlines.vtk: $streamlines_vtk_count"
echo "Subjects with streamlines_MNI.vtk: $streamlines_vtk_count_MNI"
echo "Subjects with labels_aparc+aseg.txt: $labels_txt1_count"
echo "Subjects with labels_aparc.a2009s+aseg.txt: $labels_txt2_count"
echo "Subjects with connectome.csv: $connectome_matrix_png_count"
echo " "
echo " "
echo "Subjects with labels_encoded.txt: $labels_encoded_txt_count"
echo " "

# Output file containing subject IDs with .vtk and .txt files
echo "Subject IDs with output folder containing .vtk and .txt files are saved in $output_file."
