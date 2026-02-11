#!/bin/bash
#==============================================================================
# TRACTOGRAPHY PROCESSING SCRIPT
#==============================================================================
# 
# If you downloaded HCP zip files, consider restructuring first using
# data/HCP_restructure_subjects.sh to match the expected directory layout.
#
# This script performs complete diffusion MRI tractography processing pipeline
#
# USAGE:
#   bash tractography.sh <SUBJECT_ID_OR_FILE> <DATA_DIR> [NUM_JOBS] [THREADS_PER_JOB]
#
# PARAMETERS:
#   SUBJECT_ID_OR_FILE - Either:
#                        - A specific subject ID (e.g., "100206")
#                        - "all" to process all subjects in the data directory
#                        - Path to a text file containing subject IDs (one per line)
#   DATA_DIR           - Absolute path to the directory containing subject folders
#   NUM_JOBS           - (Optional) Number of parallel jobs (subjects to process simultaneously)
#                        Default: 1 (sequential processing)
#                        Recommended: CPU_cores / THREADS_PER_JOB (e.g., 64/8 = 8 jobs)
#   THREADS_PER_JOB    - (Optional) Number of threads per subject/job
#                        Default: 8
#                        Recommended: 4-16 depending on available cores
#
# EXAMPLES:
#   # Process a single subject with 8 threads
#   bash tractography.sh 100206 /media/volume/MV_HCP/HCP_MRtrix
#
#   # Process all subjects sequentially with 8 threads each
#   bash tractography.sh all /media/volume/MV_HCP/HCP_MRtrix
#
#   # Process subjects from a list in parallel: 8 jobs, 8 threads each (64 cores total)
#   bash tractography.sh /path/to/subjects.txt /media/volume/MV_HCP/HCP_MRtrix 8 8
#
#   # Process all subjects: 4 jobs in parallel, 16 threads each (64 cores total)
#   bash tractography.sh all /media/volume/MV_HCP/HCP_MRtrix 4 16
#
#   # Maximum parallelization: 16 jobs, 4 threads each (64 cores total)
#   bash tractography.sh subjects.txt /media/volume/MV_HCP/HCP_MRtrix 16 4
#
# EXPECTED DATA STRUCTURE:
#   DATA_DIR/
#   ├── SUBJECT_ID/
#   │   ├── dMRI/
#   │   │   ├── data.nii.gz      # Diffusion-weighted images
#   │   │   ├── bvals            # B-values file
#   │   │   └── bvecs            # B-vectors file
#   │   └── anat/
#   │       ├── T1w_acpc_dc_restore.nii.gz
#   │       ├── T1w_acpc_dc_restore_brain.nii.gz
#   │       ├── aparc+aseg.nii.gz
#   │       ├── aparc.a2009s+aseg.nii.gz
#   │       └── standard2acpc_dc.nii.gz
#
# OUTPUT:
#   Each subject will have an 'output' directory created containing:
#   - Tractography log file
#   - Streamlines in VTK format
#   - Connectome matrices (CSV format)
#   - Connectome visualization plots (PNG format)
#   - Diffusion metric-weighted connectomes
#
# REQUIREMENTS:
#   - MRtrix3 (with all tools: mrconvert, dwi2response, dwi2fod, etc.)
#   - FSL (bet, flirt)
#   - FreeSurfer (for parcellation files)
#   - Python with matplotlib, numpy, pandas
#   - GNU parallel (optional, for parallel processing)
#
# PROCESSING TIME:
#   Approximate time per subject: 45-90 minutes depending on hardware
#   Main time-consuming steps:
#   - Multi-Shell Multi-Tissue CSD: ~30-45 minutes
#   - Tractography generation: ~10-20 minutes
#   - Connectome computation: ~5-15 minutes
#
#==============================================================================

# some colors for fancy logging
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Parse command-line arguments
SUBJECT_INPUT="$1"       # e.g., 100206, 'all', or path to subject list file
DATA_DIR="$2"            # folder that contains the subjects
NUM_JOBS="${3:-1}"       # Default to 1 job (sequential)
THREADS_PER_JOB="${4:-8}" # Default to 8 threads per job

# Validate inputs
if [ -z "$SUBJECT_INPUT" ] || [ -z "$DATA_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: bash tractography.sh <SUBJECT_ID_OR_FILE> <DATA_DIR> [NUM_JOBS] [THREADS_PER_JOB]"
    echo ""
    echo "Examples:"
    echo "  # Single subject, 8 threads"
    echo "  bash tractography.sh 100206 /media/volume/MV_HCP/HCP_MRtrix"
    echo ""
    echo "  # 8 parallel jobs, 8 threads each (64 cores total)"
    echo "  bash tractography.sh subjects.txt /media/volume/MV_HCP/HCP_MRtrix 8 8"
    echo ""
    echo "  # 4 parallel jobs, 16 threads each (64 cores total)"
    echo "  bash tractography.sh all /media/volume/MV_HCP/HCP_MRtrix 4 16"
    exit 1
fi

# Calculate total cores that will be used
TOTAL_CORES=$((NUM_JOBS * THREADS_PER_JOB))

echo "================================================================================"
echo "Tractography Pipeline Configuration"
echo "================================================================================"
echo "Data directory: $DATA_DIR"
echo "Parallel jobs: $NUM_JOBS"
echo "Threads per job: $THREADS_PER_JOB"
echo "Total cores used: $TOTAL_CORES"
echo "================================================================================"
echo ""

# Set threading parameter for MRtrix commands
threading="-nthreads $THREADS_PER_JOB"

process_subject() {
    local subject_id=$1
    local data_dir=$2

    start_time=$(date +%s)  # Start time

    dmri_dir="${data_dir}/${subject_id}/dMRI"
    anat_dir="${data_dir}/${subject_id}/anat"

    # Create a output directory, skip patient if already created
    output_dir="${data_dir}/${subject_id}/output"
    if [ ! -d "${output_dir}" ]; then
        mkdir -p "${output_dir}"
    else
        echo -e "${RED}[INFO]${NC} `date`: Skipping ${subject_id}, output directory already exists."
        # return #! uncomment to skip subjects with existing output
    fi

    # If any FA-weighted connectome already exists for the common parcellations, skip this subject
    # existing="${output_dir}/connectome_matrix_FA_mean_aparc+aseg.csv"
    existing="${output_dir}/connectome_matrix_SIFT_sum_aparc+aseg.csv"
    if [ -f "${existing}" ]; then
        echo -e "${RED}[INFO]${NC} `date`: Skipping ${subject_id}, connectome already exists: ${existing}"
        return
    fi

    log_file="${data_dir}/${subject_id}/output/tractography_log.txt"

    echo -e "${GREEN}[INFO]${NC} `date`: Starting tractography for: ${subject_id}" | tee -a "${log_file}"

    # First convert the initial diffusion image to .mif (~10sec)
    dwi_mif="${dmri_dir}/dwi.mif"
    if [ ! -f ${dwi_mif} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting dwi image to mif" | tee -a "${log_file}"
        # check for eddy rotated bvecs
        mrconvert "${dmri_dir}/data.nii.gz" "${dwi_mif}" \
                -fslgrad "${dmri_dir}/bvecs" "${dmri_dir}/bvals" \
                -datatype float32 -strides 0,0,0,1 ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # Then, extract mean B0 image (~1sec)
    dwi_meanbzero="${dmri_dir}/dwi_meanbzero.mif"
    dwi_meanbzero_nii="${dmri_dir}/dwi_meanbzero.nii.gz"
    if [ ! -f ${dwi_meanbzero} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Extracting mean B0 image" | tee -a "${log_file}"

        # extract mean b0
        dwiextract ${threading} -info "${dwi_mif}" -bzero - | mrmath ${threading} -info - mean -axis 3 "${dwi_meanbzero}" 2>&1
        mrconvert "${dwi_meanbzero}" "${dwi_meanbzero_nii}" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # Then, create a dwi brain mask (the provided bedpostX mask is not that accurate) (~2sec)
    dwi_meanbzero_brain="${dmri_dir}/dwi_meanbzero_brain.nii.gz"
    dwi_meanbzero_brain_mask="${dmri_dir}/dwi_meanbzero_brain_mask.nii.gz"
    if [ ! -f ${dwi_meanbzero_brain_mask} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Computing dwi brain mask" | tee -a "${log_file}"

        # Approach 2: using FSL BET (check https://github.com/sina-mansour/UKB-connectomics/commit/463b6553b5acd63f14a45ef7120145998e0a5139)

        # skull stripping to get a mask
        bet "${dwi_meanbzero_nii}" "${dwi_meanbzero_brain}" -m -R -f 0.2 -g -0.05 2>&1 | tee -a "${log_file}"
    fi

    #################################################################
    ############# CONSTRAINED SPHERICAL DECONVOLUTION ###############
    #################################################################
    start_time_csp=$(date +%s)

    # Estimate the response function using the dhollander method (~4min)
    wm_txt="${dmri_dir}/wm.txt"
    gm_txt="${dmri_dir}/gm.txt"
    csf_txt="${dmri_dir}/csf.txt"
    if [ ! -f ${wm_txt} ] || [ ! -f ${gm_txt} ] || [ ! -f ${csf_txt} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Estimation of response function using dhollander" | tee -a "${log_file}"
        dwi2response dhollander "${dwi_mif}" "${wm_txt}" "${gm_txt}" "${csf_txt}" \
                                -voxels "${dmri_dir}/voxels.mif" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution (~33min)
    wm_fod="${dmri_dir}/wmfod.mif"
    gm_fod="${dmri_dir}/gmfod.mif"
    csf_fod="${dmri_dir}/csffod.mif"
    dwi_mask_dilated="${dmri_dir}/dwi_meanbzero_brain_mask_dilated_2.nii.gz"
    if [ ! -f ${wm_fod} ] || [ ! -f ${gm_fod} ] || [ ! -f ${csf_fod} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution" | tee -a "${log_file}"
        
        # First, creating a dilated brain mask (https://github.com/sina-mansour/UKB-connectomics/issues/4)
        maskfilter -npass 2 "${dwi_meanbzero_brain_mask}" dilate "${dwi_mask_dilated}" ${threading} -info 2>&1 | tee -a "${log_file}"

        # Now, perfoming CSD with the dilated mask
        dwi2fod msmt_csd "${dwi_mif}" -mask "${dwi_mask_dilated}" "${wm_txt}" "${wm_fod}" \
                "${gm_txt}" "${gm_fod}" "${csf_txt}" "${csf_fod}" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # mtnormalise to perform multi-tissue log-domain intensity normalisation (~5sec)
    wm_fod_norm="${dmri_dir}/wmfod_norm.mif"
    gm_fod_norm="${dmri_dir}/gmfod_norm.mif"
    csf_fod_norm="${dmri_dir}/csffod_norm.mif"
    if [ ! -f ${wm_fod_norm} ] || [ ! -f ${gm_fod_norm} ] || [ ! -f ${csf_fod_norm} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running multi-tissue log-domain intensity normalisation" | tee -a "${log_file}"
        
        # First, creating an eroded brain mask (https://github.com/sina-mansour/UKB-connectomics/issues/5)
        maskfilter -npass 2 "${dwi_meanbzero_brain_mask}" erode "${dmri_dir}/dwi_meanbzero_brain_mask_eroded_2.nii.gz" ${threading} -info 2>&1 | tee -a "${log_file}"

        # Now, perfoming mtnormalise
        mtnormalise "${wm_fod}" "${wm_fod_norm}" "${gm_fod}" "${gm_fod_norm}" "${csf_fod}" \
                    "${csf_fod_norm}" -mask "${dmri_dir}/dwi_meanbzero_brain_mask_eroded_2.nii.gz" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # create a combined fod image for visualization
    vf_mif="${dmri_dir}/vf.mif"
    if [ ! -f ${vf_mif} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Generating a visualization file from normalized FODs" | tee -a "${log_file}"
        mrconvert ${threading} -info -coord 3 0 "${wm_fod_norm}" - | mrcat "${csf_fod_norm}" "${gm_fod_norm}" - "${vf_mif}" 2>&1 | tee -a "${log_file}"
    fi

    #################################################################
    ################# DIFFUSION TENSOR IMAGING ######################
    #################################################################
    start_time_dti=$(date +%s)

    # Compute DTI tensor and FA map (~30sec)
    dt_mif="${dmri_dir}/dt.mif"
    fa_mif="${dmri_dir}/fa.mif"
    md_mif="${dmri_dir}/md.mif"
    ad_mif="${dmri_dir}/ad.mif"
    rd_mif="${dmri_dir}/rd.mif"
    
    if [ ! -f ${fa_mif} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Computing diffusion tensor and FA map" | tee -a "${log_file}"
        
        # Fit diffusion tensor model
        dwi2tensor "${dwi_mif}" "${dt_mif}" -mask "${dwi_meanbzero_brain_mask}" ${threading} -info 2>&1 | tee -a "${log_file}"
        
        # Extract tensor-derived metrics (using separate commands for each metric)
        echo -e "${GREEN}[INFO]${NC} `date`: Extracting FA (Fractional Anisotropy)" | tee -a "${log_file}"
        tensor2metric "${dt_mif}" -fa "${fa_mif}" ${threading} -info 2>&1 | tee -a "${log_file}"
        
        echo -e "${GREEN}[INFO]${NC} `date`: Extracting MD (Mean Diffusivity)" | tee -a "${log_file}"  
        tensor2metric "${dt_mif}" -adc "${md_mif}" ${threading} -info 2>&1 | tee -a "${log_file}"
        
        echo -e "${GREEN}[INFO]${NC} `date`: Extracting AD (Axial Diffusivity)" | tee -a "${log_file}"
        tensor2metric "${dt_mif}" -ad "${ad_mif}" ${threading} -info 2>&1 | tee -a "${log_file}"
        
        echo -e "${GREEN}[INFO]${NC} `date`: Extracting RD (Radial Diffusivity)" | tee -a "${log_file}"
        tensor2metric "${dt_mif}" -rd "${rd_mif}" ${threading} -info 2>&1 | tee -a "${log_file}"
        
        # Convert FA to NIfTI for compatibility
        mrconvert "${fa_mif}" "${dmri_dir}/fa.nii.gz" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    end_time_dti=$(date +%s)  # End time
    elapsed_time_dti=$((end_time_dti - start_time_dti))

    end_time_csp=$(date +%s)  # End time
    elapsed_time_csp=$((end_time_csp - start_time_csp))
    

    #################################################################
    ################### CREATING TISSUE BOUNDARY ####################
    #################################################################
    start_time_tb=$(date +%s)


    # Create a mask of white matter gray matter interface using 5 tissue type segmentation (~70sec)
    T1_brain_nii="${anat_dir}/T1w_acpc_dc_restore_brain.nii.gz"
    T1_brain="${anat_dir}/T1w_acpc_dc_restore_brain.mif"
    if [ ! -f ${T1_brain} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting T1 brain image to mif" | tee -a "${log_file}"
        mrconvert "${T1_brain_nii}" "${T1_brain}" 2>&1 | tee -a "${log_file}"
    fi

    T1_nii="${anat_dir}/T1w_acpc_dc_restore.nii.gz"
    T1="${anat_dir}/T1w_acpc_dc_restore.mif"
    if [ ! -f ${T1} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting T1 image to mif" | tee -a "${log_file}"
        mrconvert "${T1_nii}" "${T1}" 2>&1 | tee -a "${log_file}"
    fi

    parcellation_nii="${anat_dir}/aparc+aseg.nii.gz"
    parcellation="${anat_dir}/aparc+aseg.mif"
    if [ ! -f ${parcellation} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting parcellation image to mif" | tee -a "${log_file}"
        mrconvert "${parcellation_nii}" "${parcellation}" 2>&1 | tee -a "${log_file}"
    fi

    seg_5tt_T1="${dmri_dir}/seg_5tt_T1.mif"
    seg_5tt="${dmri_dir}/seg_5tt.mif"

    T1_brain_dwi="${dmri_dir}/T1_brain_dwi.mif"
    T1_brain_mask="${anat_dir}/T1_brain_mask.nii.gz"

    gmwm_seed_T1="${dmri_dir}/gmwm_seed_T1.mif"
    gmwm_seed="${dmri_dir}/gmwm_seed.mif"
    transform_DWI_T1_FSL="${dmri_dir}/diff2struct_fsl.txt"
    transform_DWI_T1="${dmri_dir}/diff2struct_mrtrix.txt"

    if [ ! -f ${gmwm_seed} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running 5ttgen to get gray matter white matter interface mask" | tee -a "${log_file}"
        # First create the 5tt image
        # Option 1: generate the 5TT image based on a FreeSurfer parcellation image
        # 5ttgen freesurfer "${parcellation}" "${seg_5tt_T1}" -nocrop ${threading} -info 2>&1 | tee -a "${log_file}"
        
        # Option 2: generate the 5TT image based on a T1 image with FSL
        5ttgen fsl ${threading} "${T1_brain}" "${seg_5tt_T1}" -premasked -info 2>&1 | tee -a "${log_file}"
        # 5ttgen fsl "${T1_brain_nii}" "${seg_5tt_T1}" -premasked ${threading} -info
        # 5ttgen fsl "${T1_nii}" "${seg_5tt_T1}" ${threading} -nocrop -info  
        # 5ttgen fsl "${T1}" "${seg_5tt_T1}" ${threading} -nocrop ${threading} -info  

        # Next generate the boundary ribbon
        5tt2gmwmi ${threading} "${seg_5tt_T1}" "${gmwm_seed_T1}" -info 2>&1 | tee -a "${log_file}"

        # Coregistering the Diffusion and Anatomical Images
        # Perform rigid body registration
        flirt -in "${dwi_meanbzero_brain}" -ref "${T1_brain_nii}" \
            -cost normmi -dof 6 -omat "${transform_DWI_T1_FSL}" 2>&1 | tee -a "${log_file}"
        transformconvert "${transform_DWI_T1_FSL}" "${dwi_meanbzero_brain}" \
                        "${T1_brain}" flirt_import "${transform_DWI_T1}" 2>&1 | tee -a "${log_file}"

        # Perform transformation of the boundary ribbon from T1 to DWI space
        mrtransform "${seg_5tt_T1}" "${seg_5tt}" -linear "${transform_DWI_T1}" -inverse ${threading} -info 2>&1 | tee -a "${log_file}"
        mrtransform "${T1_brain}" "${T1_brain_dwi}" -linear "${transform_DWI_T1}" -inverse ${threading} -info 2>&1 | tee -a "${log_file}"
        mrtransform "${gmwm_seed_T1}" "${gmwm_seed}" -linear "${transform_DWI_T1}" -inverse ${threading} -info 2>&1 | tee -a "${log_file}"
        
        # Visualize result
        # mrview T1_brain_dwi.mif -overlay.load gmwm_seed.mif -overlay.colourmap 2 -overlay.load gmwm_seed_T1.mif -overlay.colourmap 1
        # mrview ../anat/T1_brain.mif -overlay.load gmwm_seed.mif -overlay.colourmap 2 -overlay.load gmwm_seed_T1.mif -overlay.colourmap 1
        # mrview dwi_meanbzero.mif -overlay.load gmwm_seed.mif -overlay.colourmap 2 -overlay.load gmwm_seed_T1.mif -overlay.colourmap 1
        # mrview dwi_meanbzero.mif -overlay.load seg_5tt.mif -overlay.colourmap 2 -overlay.load seg_5tt_T1.mif -overlay.colourmap 1
    fi

    end_time_tb=$(date +%s)  # End time
    elapsed_time_tb=$((end_time_tb - start_time_tb))



    #################################################################
    ########################## STREAMLINES ##########################
    #################################################################
    start_time_tractography=$(date +%s)
    streamlines=10M
    # Create streamlines
    tracts="${dmri_dir}/tracts_${streamlines}.tck"
    tractstats="${dmri_dir}/stats/${subject_id}_tracts_${streamlines}_stats.json"
    mkdir -p "${dmri_dir}/stats"
    if [ ! -f ${tracts} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running probabilistic tractography" | tee -a "${log_file}"
        # Seed until #streamlines reached
        # tckgen -seed_gmwmi "${gmwm_seed}" -act "${seg_5tt}" -select "${streamlines}" \
        #                     -maxlength 250 -cutoff 0.1 ${threading} "${wm_fod_norm}" "${tracts}" -power 0.5 \
        #                     -info -samples 3  2>&1 | tee -a "${log_file}" #-output_stats "${tractstats}"
        # Seed #streamlines
        tckgen -seed_gmwmi "${gmwm_seed}" -act "${seg_5tt}" -seeds "${streamlines}" \
                            -maxlength 250 -cutoff 0.1 ${threading} "${wm_fod_norm}" "${tracts}" -power 0.5 \
                            -info -samples 3  2>&1 | tee -a "${log_file}" #-output_stats "${tractstats}"
        # Visualize result
        # tckedit "${tracts}" -number 200k "${dmri_dir}/tracts_200k.tck"
        # mrview dwi_meanbzero.mif -tractography.load smallertracts_200k.tck
        # mrview anat/T1w_acpc_dc_restore_brain.nii.gz -tractography.load dMRI/tracts_200k.tck
        # mrview 103818/anat/T1w_acpc_dc_restore_brain.nii.gz -tractography.load 103818/dMRI/tracts_200k.tck
    fi

    tracts_MNI="${dmri_dir}/tracts_${streamlines}_MNI.tck"
    if [ ! -f ${tracts_MNI} ]; then
        # Prepare deformation field file
        mrconvert ${anat_dir}/standard2acpc_dc.nii.gz ${dmri_dir}/tmp-[].nii -force
        mv ${dmri_dir}/tmp-0.nii ${dmri_dir}/x.nii
        mrcalc ${dmri_dir}/x.nii -neg ${dmri_dir}/tmp-0.nii -force
        warpconvert ${dmri_dir}/tmp-[].nii displacement2deformation ${dmri_dir}/acpc2MNI_mrtrix.nii.gz -force
        rm ${dmri_dir}/x.nii ${dmri_dir}/tmp-?.nii

        # Transform tracts to MNI space
        tcktransform "${tracts}" \
                    "${dmri_dir}/acpc2MNI_mrtrix.nii.gz" \
                    "${tracts_MNI}"
        # Generate subsample for visualization
        # tckedit "${tracts_MNI}" -number 200k "${dmri_dir}/tracts_200k_MNI.tck"
        # mrview anat/T1w_restore_brain.nii.gz -tractography.load dMRI/tracts_200k_MNI.tck
    fi

    end_time_tractography=$(date +%s)  # End time
    elapsed_time_tractography=$((end_time_tractography - start_time_tractography))

    #!################################################################# 
    #!###################### SIFT2 STREAMLINE WEIGHTS #################
    #!#################################################################
    start_time_sift=$(date +%s)
    sift_weights="${dmri_dir}/sift2_weights.txt"
    if [ ! -f ${sift_weights} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running SIFT2 to calculate streamline weights" | tee -a "${log_file}"
        tcksift2 -act "${seg_5tt}" -out_mu "${output_dir}/sift_mu.txt" \
                 ${threading} -info \
                 "${tracts}" "${wm_fod_norm}" "${sift_weights}" 2>&1 | tee -a "${log_file}"
    fi
    end_time_sift=$(date +%s)
    elapsed_time_sift=$((end_time_sift - start_time_sift))

    # Calculate and save length of each streamline
    streamline_lengths_file="${output_dir}/streamline_lengths_${streamlines}.txt"
    if [ ! -f ${streamline_lengths_file} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Calculating and saving length of each streamline" | tee -a "${log_file}"
        
        tckstats -dump "${streamline_lengths_file}" "${tracts}" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    #################################################################
    ################## MAP STRUCTURAL CONNECTIVITY ##################
    #################################################################

    # Write tracts to vtk file
    tracts_vtk=${output_dir}/streamlines_${streamlines}.vtk
    if [ ! -f ${tracts_vtk} ]; then
        tckconvert -binary ${tracts} ${tracts_vtk} 2>&1 | tee -a "${log_file}" # might give error as -binary became an option in mrtrix 3.0.4 (download with conda install -c mrtrix3 mrtrix3)
    fi
    tracts_vtk_MNI=${output_dir}/streamlines_${streamlines}_MNI.vtk
    if [ ! -f ${tracts_vtk_MNI} ]; then
        tckconvert -binary ${tracts_MNI} ${tracts_vtk_MNI} 2>&1 | tee -a "${log_file}" 
    fi

    # Define both parcellations
    parcellations=("aparc+aseg" "aparc.a2009s+aseg")

    # Loop over both parcellations
    for parc in "${parcellations[@]}"; do

        # Define paths for the current parcellation
        parcellation_nii="${anat_dir}/${parc}.nii.gz"
        parcellation="${anat_dir}/${parc}.mif"
        parcellation_converted="${anat_dir}/${parc}_mrtrix.mif"
        connectome_matrix="${output_dir}/connectome_matrix_${parc}.csv"

        # Convert parcellation image to mif if not already converted
        if [ ! -f ${parcellation} ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Converting ${parc} parcellation image to mif" | tee -a "${log_file}"
            mrconvert "${parcellation_nii}" "${parcellation}" 2>&1 | tee -a "${log_file}"
        fi

        # Convert Freesurfer labels to MRtrix if not already converted
        if [ ! -f ${parcellation_converted} ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Converting Freesurfer labels for ${parc} to MRtrix" | tee -a "${log_file}"
            # labelconvert "${parcellation}" $FREESURFER_HOME/FreeSurferColorLUT.txt $MRTRIX3_HOME/share/mrtrix3/labelconvert/fs_default.txt "${parcellation_converted}" 2>&1 | tee -a "${log_file}"
            labelconvert "${parcellation}" ./txt_files/FreeSurferColorLUT.txt ./txt_files/fs_${parc}.txt "${parcellation_converted}" 2>&1 | tee -a "${log_file}"
        fi

        # Generate connectome matrix if not already generated
        if [ ! -f ${connectome_matrix} ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Computing connectome matrix from streamline count for ${parc}" | tee -a "${log_file}"

            # Record the start time
            start_time=$(date +%s)

            tck2connectome ${threading} -info -symmetric \
                            "${tracts}" "${parcellation_converted}" "${connectome_matrix}" \
                            -out_assignments ${output_dir}/labels_${streamlines}_${parc}.txt 2>&1 | tee -a "${log_file}"

            # Record the end time
            end_time=$(date +%s)

            # Calculate the elapsed time and log it
            elapsed_time=$((end_time - start_time))
            echo -e "${GREEN}[INFO]${NC} `date`: tck2connectome completed for ${parc} in ${elapsed_time} seconds" | tee -a "${log_file}"

            # Generate the connectome matrix plot
            python plot_connectome.py "${connectome_matrix}" "${output_dir}/connectome_matrix_${streamlines}_${parc}.png" "Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"
        fi

        

        # Generate FA-weighted connectome matrix
        connectome_matrix_fa_mean="${output_dir}/connectome_matrix_FA_mean_${parc}.csv"
        connectome_matrix_md_mean="${output_dir}/connectome_matrix_MD_mean_${parc}.csv"
        connectome_matrix_ad_mean="${output_dir}/connectome_matrix_AD_mean_${parc}.csv"
        connectome_matrix_rd_mean="${output_dir}/connectome_matrix_RD_mean_${parc}.csv"
        #!!!!!!!!!
        connectome_matrix_sift_sum="${output_dir}/connectome_matrix_SIFT_sum_${parc}.csv"
        
        if [ ! -f ${connectome_matrix_fa_mean} ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Computing diffusion-weighted connectome matrices for ${parc}" | tee -a "${log_file}"

            # Record the start time
            start_time=$(date +%s)

            # Sample diffusion metric values along streamlines
            echo -e "${GREEN}[INFO]${NC} `date`: Sampling mean FA values per streamline" | tee -a "${log_file}"
            tcksample "${tracts}" "${fa_mif}" "${dmri_dir}/mean_fa_per_streamline.txt" -stat_tck mean ${threading} -info 2>&1 | tee -a "${log_file}"
            
            echo -e "${GREEN}[INFO]${NC} `date`: Sampling mean MD values per streamline" | tee -a "${log_file}"
            tcksample "${tracts}" "${md_mif}" "${dmri_dir}/mean_md_per_streamline.txt" -stat_tck mean ${threading} -info 2>&1 | tee -a "${log_file}"
            
            echo -e "${GREEN}[INFO]${NC} `date`: Sampling mean AD values per streamline" | tee -a "${log_file}"
            tcksample "${tracts}" "${ad_mif}" "${dmri_dir}/mean_ad_per_streamline.txt" -stat_tck mean ${threading} -info 2>&1 | tee -a "${log_file}"
            
            echo -e "${GREEN}[INFO]${NC} `date`: Sampling mean RD values per streamline" | tee -a "${log_file}"
            tcksample "${tracts}" "${rd_mif}" "${dmri_dir}/mean_rd_per_streamline.txt" -stat_tck mean ${threading} -info 2>&1 | tee -a "${log_file}"
            
            # Create diffusion metric-weighted connectomes
            echo -e "${GREEN}[INFO]${NC} `date`: Generating FA-weighted connectome" | tee -a "${log_file}"
            tck2connectome ${threading} -info -symmetric \
                            "${tracts}" "${parcellation_converted}" "${connectome_matrix_fa_mean}" \
                            -scale_file "${dmri_dir}/mean_fa_per_streamline.txt" -stat_edge mean 2>&1 | tee -a "${log_file}"

            # Create MD-weighted connectome (mean diffusivity)
            echo -e "${GREEN}[INFO]${NC} `date`: Generating MD-weighted connectome" | tee -a "${log_file}"
            tck2connectome ${threading} -info -symmetric \
                            "${tracts}" "${parcellation_converted}" "${connectome_matrix_md_mean}" \
                            -scale_file "${dmri_dir}/mean_md_per_streamline.txt" -stat_edge mean 2>&1 | tee -a "${log_file}"

            # Create AD-weighted connectome (axial diffusivity)
            echo -e "${GREEN}[INFO]${NC} `date`: Generating AD-weighted connectome" | tee -a "${log_file}"
            tck2connectome ${threading} -info -symmetric \
                            "${tracts}" "${parcellation_converted}" "${connectome_matrix_ad_mean}" \
                            -scale_file "${dmri_dir}/mean_ad_per_streamline.txt" -stat_edge mean 2>&1 | tee -a "${log_file}"

            # Create RD-weighted connectome (radial diffusivity)
            echo -e "${GREEN}[INFO]${NC} `date`: Generating RD-weighted connectome" | tee -a "${log_file}"
            tck2connectome ${threading} -info -symmetric \
                            "${tracts}" "${parcellation_converted}" "${connectome_matrix_rd_mean}" \
                            -scale_file "${dmri_dir}/mean_rd_per_streamline.txt" -stat_edge mean 2>&1 | tee -a "${log_file}"
            #!!!!!!!!!!!!!
            echo -e "${GREEN}[INFO]${NC} `date`: Generating SIFT2-weighted connectome" | tee -a "${log_file}"
            tck2connectome ${threading} -info -symmetric \
                           "${tracts}" "${parcellation_converted}" "${connectome_matrix_sift_sum}" \
                           -scale_file "${sift_weights}" -stat_edge sum 2>&1 | tee -a "${log_file}"
            
            # Record the end time
            end_time=$(date +%s)

            # Calculate the elapsed time and log it
            elapsed_time=$((end_time - start_time))
            echo -e "${GREEN}[INFO]${NC} `date`: Diffusion-weighted connectome generation completed for ${parc} in ${elapsed_time} seconds" | tee -a "${log_file}"

            # Generate the diffusion-weighted connectome matrix plots
            python plot_connectome.py "${connectome_matrix_fa_mean}" "${output_dir}/connectome_matrix_FA_mean_${streamlines}_${parc}.png" "FA-weighted (mean) Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"
            python plot_connectome.py "${connectome_matrix_md_mean}" "${output_dir}/connectome_matrix_MD_mean_${streamlines}_${parc}.png" "MD-weighted (mean) Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"
            python plot_connectome.py "${connectome_matrix_ad_mean}" "${output_dir}/connectome_matrix_AD_mean_${streamlines}_${parc}.png" "AD-weighted (mean) Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"
            python plot_connectome.py "${connectome_matrix_rd_mean}" "${output_dir}/connectome_matrix_RD_mean_${streamlines}_${parc}.png" "RD-weighted (mean) Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"
            python plot_connectome.py "${connectome_matrix_rd_mean}" "${output_dir}/connectome_matrix_RD_mean_${streamlines}_${parc}.png" "RD-weighted (mean) Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"
            #!!!!!!!!!!!
            python plot_connectome.py "${connectome_matrix_sift_sum}" "${output_dir}/connectome_matrix_SIFT_sum_${streamlines}_${parc}.png" "SIFT2-weighted Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"

        fi



    done

    echo -e "${GREEN}[INFO]${NC} `date`: Finished tractography for: ${subject_id}" | tee -a "${log_file}"
    
    end_time=$(date +%s)  # End time
    elapsed_time=$((end_time - start_time))

    echo ""
    echo -e "${GREEN}[INFO]${NC} `date`: Constrained Spherical Deconvolution took: ${elapsed_time_csp} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Diffusion Tensor Imaging took: ${elapsed_time_dti} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Generating tissue boundary took: ${elapsed_time_tb} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Tractography took: ${elapsed_time_tractography} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: SIFT2 weighting took: ${elapsed_time_sift} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Finished processing ${subject_id}. Total time: ${elapsed_time} seconds." | tee -a "${log_file}"

    # Call the Python script to clean the log file
    python3 ./clean_log.py "${log_file}"
    echo "Log file cleaned!"

    # Remove intermediate tract files to save space
    for f in "${tracts}" "${tracts_MNI}" "${tracts_vtk}"; do # "${tracts_vtk_MNI}"; do
        if [ -n "${f}" ] && [ -f "${f}" ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Removing ${f}" | tee -a "${log_file}"
            rm -f "${f}" 2>&1 | tee -a "${log_file}"
        fi
    done
}


if [ "${SUBJECT_INPUT}" == "all" ]; then
    # Get a list of all subjects in the data directory
    subjects=$(ls "${DATA_DIR}" | grep -E '^[0-9]{6}$')
    subject_count=$(echo "${subjects}" | wc -l)
    
    echo -e "${GREEN}[INFO]${NC} `date`: Processing ${subject_count} subjects from data directory: ${DATA_DIR}"
    echo -e "${GREEN}[INFO]${NC} `date`: Subject list: $(echo ${subjects} | tr '\n' ' ')"
    
    if [ "$NUM_JOBS" -gt 1 ]; then
        # Parallel processing using bash job control
        echo -e "${GREEN}[INFO]${NC} `date`: Running ${NUM_JOBS} parallel jobs with ${THREADS_PER_JOB} threads each"
        
        # Check if GNU parallel is available
        if command -v parallel &> /dev/null; then
            # Use GNU parallel if available
            export -f process_subject
            export DATA_DIR threading RED GREEN NC
            echo "${subjects}" | parallel -j "$NUM_JOBS" --line-buffer "process_subject {} $DATA_DIR"
        else
            # Fallback to bash job control
            echo -e "${GREEN}[INFO]${NC} Note: GNU parallel not found, using bash job control"
            
            running_jobs=0
            current=1
            
            for subject in ${subjects}; do
                # Wait if we've reached the max number of parallel jobs
                while [ $running_jobs -ge $NUM_JOBS ]; do
                    wait -n  # Wait for any job to finish
                    running_jobs=$((running_jobs - 1))
                done
                
                # Start new job in background
                echo -e "${GREEN}[INFO]${NC} `date`: Starting processing for subject: ${subject} (${current}/${subject_count})"
                process_subject "${subject}" "${DATA_DIR}" &
                running_jobs=$((running_jobs + 1))
                current=$((current + 1))
            done
            
            # Wait for all remaining jobs to complete
            wait
            echo -e "${GREEN}[INFO]${NC} `date`: All parallel jobs completed"
        fi
    else
        # Sequential processing
        current=1
        for subject in ${subjects}; do
            echo -e "${GREEN}[INFO]${NC} `date`: Starting processing for subject: ${subject} (${current}/${subject_count})"
            process_subject "${subject}" "${DATA_DIR}"
            echo -e "${GREEN}[INFO]${NC} `date`: Completed processing for subject: ${subject} (${current}/${subject_count})"
            echo "----------------------------------------"
            ((current++))
        done
    fi
    
elif [ -f "${SUBJECT_INPUT}" ]; then
    # Read subjects from the specified text file
    subjects_file="${SUBJECT_INPUT}"
    
    # Read subjects from file, removing any potential whitespace
    subjects=$(cat "${subjects_file}" | tr -d '\r' | grep -v '^$')
    subject_count=$(echo "${subjects}" | wc -l)
    
    echo -e "${GREEN}[INFO]${NC} `date`: Processing ${subject_count} subjects from file: ${subjects_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Subject list: $(echo ${subjects} | tr '\n' ' ')"
    
    if [ "$NUM_JOBS" -gt 1 ]; then
        # Parallel processing using bash job control
        echo -e "${GREEN}[INFO]${NC} `date`: Running ${NUM_JOBS} parallel jobs with ${THREADS_PER_JOB} threads each"
        
        # Check if GNU parallel is available
        if command -v parallel &> /dev/null; then
            # Use GNU parallel if available
            export -f process_subject
            export DATA_DIR threading RED GREEN NC
            echo "${subjects}" | parallel -j "$NUM_JOBS" --line-buffer "process_subject {} $DATA_DIR"
        else
            # Fallback to bash job control
            echo -e "${GREEN}[INFO]${NC} Note: GNU parallel not found, using bash job control"
            
            running_jobs=0
            current=1
            
            for subject in ${subjects}; do
                # Wait if we've reached the max number of parallel jobs
                while [ $running_jobs -ge $NUM_JOBS ]; do
                    wait -n  # Wait for any job to finish
                    running_jobs=$((running_jobs - 1))
                done
                
                # Start new job in background
                echo -e "${GREEN}[INFO]${NC} `date`: Starting processing for subject: ${subject} (${current}/${subject_count})"
                process_subject "${subject}" "${DATA_DIR}" &
                running_jobs=$((running_jobs + 1))
                current=$((current + 1))
            done
            
            # Wait for all remaining jobs to complete
            wait
            echo -e "${GREEN}[INFO]${NC} `date`: All parallel jobs completed"
        fi
    else
        # Sequential processing
        current=1
        for subject in ${subjects}; do
            echo -e "${GREEN}[INFO]${NC} `date`: Starting processing for subject: ${subject} (${current}/${subject_count})"
            process_subject "${subject}" "${DATA_DIR}"
            echo -e "${GREEN}[INFO]${NC} `date`: Completed processing for subject: ${subject} (${current}/${subject_count})"
            echo "----------------------------------------"
            ((current++))
        done
    fi
    
else
    # Process single subject
    process_subject "${SUBJECT_INPUT}" "${DATA_DIR}"
fi