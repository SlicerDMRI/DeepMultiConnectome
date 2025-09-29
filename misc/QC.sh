#!/bin/bash

# data_dir=$1
subject_dir=$1

# for subject_dir in ${data_dir}/*/ ; do
# if [ -d "${subject_dir}/output" ]; then
subject_id=$(basename ${subject_dir})
echo "Quality control for subject ${subject_id}"
# echo "${subject_dir}"

dwi_image="${subject_dir}/dMRI/dwi_meanbzero.mif"
overlay_image="${subject_dir}/dMRI/gmwm_seed.mif"
CQ_dir="${subject_dir}/output/QC/"
prefix_gmwm="${subject_dir}_B0_GWMW_"
prefix_tck="${subject_dir}_B0_200K_streamlines_"



# Ensure the QC directory exists
if [ ! -d ${CQ_dir} ]; then
    mkdir -p ${CQ_dir}
fi


# Capture view with GMWM boundary
echo "View GMWM boundary"
mrview "${dwi_image}" -overlay.load "${overlay_image}" -overlay.colourmap 1 -mode 2 -focus 20,-10,10 -capture.folder "${CQ_dir}/GWMW" -noannot -capture.grab -exit

# Capture view with subsample of the tracts
echo "View subsample of tracts"
if [ ! -f "${subject_dir}/dMRI/tracts_subsample_200K.tck" ]; then
    tckedit "${subject_dir}/dMRI/tracts_10M.tck" -number 200K "${subject_dir}/dMRI/tracts_subsample_200K.tck"
fi

mrview "${dwi_image}" -tractography.load "${subject_dir}/dMRI/tracts_subsample_200K.tck" -mode 2 -focus 20,-10,10 -capture.folder "${CQ_dir}/tracts" -noannot -capture.grab -exit

# Capture view with GMWM boundary
# echo "View GMWM boundary"
# mrview ${dwi_image} -overlay.load ${overlay_image} -overlay.colourmap 1 -mode 2 -focus 20,-10,10 -capture.folder ${CQ_dir} -capture.prefix ${prefix_gmwm} -capture.grab -exit $ #  -info > ${CQ_dir}/gmwm_capture.log 2>&1

# # Sleep to ensure the command completes
# sleep 5
# wait $!

# # Capture view with subsample of the tracts
# echo "View subsample of tracts"
# if [ ! -f ${subject_dir}/dMRI/tracts_subsample_200K.tck ]; then
#     tckedit ${subject_dir}/dMRI/tracts_10M.tck -number 200K ${subject_dir}/dMRI/tracts_subsample_200K.tck
# fi

# # mrview ${dwi_image} -tractography.load ${subject_dir}/dMRI/tracts_subsample_200K.tck -mode 2 -focus 20,-10,10 -capture.folder ${CQ_dir} -capture.prefix ${prefix_tck} -capture.grab -exit #-info > ${CQ_dir}/tck_capture.log 2>&1
# mrview ${dwi_image} -tractography.load ${subject_dir}/dMRI/tracts_subsample_200K.tck -mode 2 -capture.folder ${CQ_dir} -capture.prefix ${prefix_tck} -capture.grab -exit $ # -info > ${CQ_dir}/tck_capture.log 2>&1
# sleep 5
# wait $!

# mrview ${dit}/dMRI/dwi_meanbzero.mif -tractography.load ${dit}/dMRI/tracts_subsample_200K.tck -mode 2 -focus 20,-10,10 -capture.folder ${dit}/output/QC -capture.prefix streamlines -capture.grab -exit
# fi
# done

# mrview dMRI/dwi_meanbzero.mif -overlay.load dMRI/gmwm_seed.mif -overlay.colourmap 1 -mode 2 -focus 20,-10,10 -capture.folder output/QC/GWMW -capture.prefix test_prefix_ -capture.grab -exit