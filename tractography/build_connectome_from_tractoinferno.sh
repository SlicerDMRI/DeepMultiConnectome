#!/bin/bash
#==============================================================================
# BUILD CONNECTOME FROM TRACTOINFERNO STREAMLINES
#==============================================================================
# 
# This script builds structural connectomes using pre-computed streamlines
# from TractoInferno dataset and FastSurfer parcellations
#
# USAGE:
#   bash build_connectome_from_tractoinferno.sh <SUBJECT_ID> <DATA_DIR> [THREADS]
#
# PARAMETERS:
#   SUBJECT_ID  - Subject ID (e.g., "1001")
#   DATA_DIR    - Path to organized data directory
#   THREADS     - Number of threads (default: 8)

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Parse arguments
subject_id=$1
data_dir=$2
num_threads=${3:-8}

if [ -z "$subject_id" ] || [ -z "$data_dir" ]; then
    echo "Usage: bash build_connectome_from_tractoinferno.sh <SUBJECT_ID> <DATA_DIR> [THREADS]"
    exit 1
fi

# Set threading
threading="-nthreads ${num_threads}"

# Paths
subject_dir="${data_dir}/${subject_id}"
anat_dir="${subject_dir}/anat"
dmri_dir="${subject_dir}/dMRI"
output_dir="${subject_dir}/output"
tractoinferno_dir="/media/volume/HCP_diffusion_MV/tractoinferno/derivatives/trainset/sub-${subject_id}"
tract_dir="${tractoinferno_dir}/tractography"

# Create output directory
mkdir -p "${output_dir}"

log_file="${output_dir}/connectome_build_log.txt"
echo -e "${GREEN}[INFO]${NC} $(date): Starting connectome construction for: ${subject_id}" | tee "${log_file}"

# Check if TractoInferno streamlines exist
if [ ! -d "${tract_dir}" ]; then
    echo -e "${RED}[ERROR]${NC} TractoInferno tractography directory not found: ${tract_dir}" | tee -a "${log_file}"
    exit 1
fi

# Check if FastSurfer parcellation exists
parcellation_nii="${anat_dir}/aparc+aseg.nii.gz"
if [ ! -f "${parcellation_nii}" ]; then
    echo -e "${RED}[ERROR]${NC} Parcellation not found: ${parcellation_nii}" | tee -a "${log_file}"
    exit 1
fi

# Convert parcellation to MIF
parcellation="${anat_dir}/aparc+aseg.mif"
if [ ! -f "${parcellation}" ]; then
    echo -e "${GREEN}[INFO]${NC} $(date): Converting parcellation to MIF" | tee -a "${log_file}"
    mrconvert "${parcellation_nii}" "${parcellation}" ${threading} -info 2>&1 | tee -a "${log_file}"
fi

# Convert TRK files to TCK and combine
echo -e "${GREEN}[INFO]${NC} $(date): Converting and combining TractoInferno streamlines" | tee -a "${log_file}"

combined_tck="${dmri_dir}/tractoinferno_combined.tck"
temp_tck_list="${dmri_dir}/temp_tck_list.txt"

# Convert all TRK bundles to TCK format using Python
> "${temp_tck_list}"  # Clear file
converter_script="/media/volume/HCP_diffusion_MV/DeepMultiConnectome/tractography/convert_trk_to_tck.py"
for trk_file in ${tract_dir}/*.trk; do
    bundle_name=$(basename "${trk_file}" .trk)
    tck_file="${dmri_dir}/${bundle_name}.tck"
    
    echo -e "${GREEN}[INFO]${NC} Converting: ${bundle_name}" | tee -a "${log_file}"
    python "${converter_script}" "${trk_file}" "${tck_file}" 2>&1 | tee -a "${log_file}"
    
    if [ -f "${tck_file}" ]; then
        echo "${tck_file}" >> "${temp_tck_list}"
    fi
done

# Combine all TCK files into one
echo -e "${GREEN}[INFO]${NC} $(date): Combining all streamlines" | tee -a "${log_file}"
tckedit $(cat "${temp_tck_list}") "${combined_tck}" -force ${threading} -info 2>&1 | tee -a "${log_file}"

# Get streamline count
streamline_count=$(tckinfo "${combined_tck}" | grep "count:" | awk '{print $2}')
echo -e "${GREEN}[INFO]${NC} Total streamlines: ${streamline_count}" | tee -a "${log_file}"

# Convert labels for MRtrix
echo -e "${GREEN}[INFO]${NC} $(date): Converting FreeSurfer labels for aparc+aseg to MRtrix" | tee -a "${log_file}"
parc_mrtrix="${anat_dir}/aparc+aseg_mrtrix.mif"
labelconvert "${parcellation}" \
    /media/volume/HCP_diffusion_MV/DeepMultiConnectome/tractography/txt_files/FreeSurferColorLUT.txt \
    /media/volume/HCP_diffusion_MV/DeepMultiConnectome/tractography/txt_files/fs_aparc+aseg.txt \
    "${parc_mrtrix}" -force ${threading} -info 2>&1 | tee -a "${log_file}"

# Build structural connectivity matrix
echo -e "${GREEN}[INFO]${NC} $(date): Building structural connectome" | tee -a "${log_file}"
connectome_csv="${output_dir}/connectome_matrix_aparc+aseg.csv"
connectome_assignments="${output_dir}/connectome_assignments_aparc+aseg.txt"

tck2connectome "${combined_tck}" "${parc_mrtrix}" "${connectome_csv}" \
    -assignment_radial_search 2 \
    -out_assignments "${connectome_assignments}" \
    ${threading} -info 2>&1 | tee -a "${log_file}"

echo -e "${GREEN}[INFO]${NC} $(date): Connectome saved to: ${connectome_csv}" | tee -a "${log_file}"

# Get connectome statistics
num_nodes=$(head -1 "${connectome_csv}" | tr ',' '\n' | wc -l)
num_edges=$(awk -F',' '{for(i=1;i<=NF;i++) if($i>0) count++} END {print count}' "${connectome_csv}")
echo -e "${GREEN}[INFO]${NC} Connectome: ${num_nodes} nodes, ${num_edges} non-zero edges" | tee -a "${log_file}"

# Optional: Calculate mean length per connection
echo -e "${GREEN}[INFO]${NC} $(date): Computing mean streamline length connectome" | tee -a "${log_file}"
connectome_length="${output_dir}/connectome_matrix_mean_length_aparc+aseg.csv"
tck2connectome "${combined_tck}" "${parc_mrtrix}" "${connectome_length}" \
    -scale_length -stat_edge mean \
    ${threading} -info 2>&1 | tee -a "${log_file}"

# Clean up temporary files
rm -f "${temp_tck_list}"
rm -f ${dmri_dir}/sub-${subject_id}__*.tck

echo -e "${GREEN}[INFO]${NC} $(date): Finished building connectome for ${subject_id}" | tee -a "${log_file}"
echo -e "${GREEN}[INFO]${NC} Outputs:" | tee -a "${log_file}"
echo -e "${GREEN}[INFO]${NC}   - Combined streamlines: ${combined_tck}" | tee -a "${log_file}"
echo -e "${GREEN}[INFO]${NC}   - Connectome matrix: ${connectome_csv}" | tee -a "${log_file}"
echo -e "${GREEN}[INFO]${NC}   - Mean length matrix: ${connectome_length}" | tee -a "${log_file}"
