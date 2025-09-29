#!/bin/bash

# Define the base data directory
data_dir=$1

# Create the destination directory if it doesn't exist
destination_dir="10M_streamlines"
mkdir -p "$destination_dir"

# Loop over each subject directory in the data directory
for subject_dir in ${data_dir}/*/ ; do
    # Extract the subject ID from the directory name
    subject_id=$(basename "$subject_dir")

    # Define the source file path
    source_file="${subject_dir}/output/tracts/${subject_id}_streamlines_10M.vtk"

    # Define the destination file path
    destination_file="${destination_dir}/${subject_id}.vtk"

    # Check if the source file exists
    if [ -f "$source_file" ]; then
        # Move the file to the destination directory and rename it
        cp "$source_file" "$destination_file"
        echo "Moved ${source_file} to ${destination_file}"
    else
        echo "File ${source_file} does not exist"
    fi
done
