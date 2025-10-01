#!/bin/bash

#==============================================================================
# CONNECTOME MATRIX COPY SCRIPT
#==============================================================================
# 
# This script copies connectome matrices from all subjects in the HCP_MRtrix
# directory to a centralized location for analysis.
#
# USAGE:
#   bash copy_connectomes.sh <SOURCE_DIR>
#
# PARAMETERS:
#   SOURCE_DIR  - Path to directory containing subject folders (e.g., /media/volume/MV_HCP/HCP_MRtrix)
#
# EXAMPLE:
#   bash copy_connectomes.sh /media/volume/MV_HCP/HCP_MRtrix
#
# OUTPUT:
#   - Copies connectome matrices to /media/volume/MV_HCP/HCP_all_connectomes/SUBJECT_ID/
#   - Creates missing_connectomes.txt listing subjects with missing files
#   - Creates copy_log.txt with detailed processing information
#
#==============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if source directory is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}[ERROR]${NC} Please provide the source directory path."
    echo "Usage: bash copy_connectomes.sh <SOURCE_DIR>"
    echo "Example: bash copy_connectomes.sh /media/volume/MV_HCP/HCP_MRtrix"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="/media/volume/MV_HCP/HCP_all_connectomes"
LOG_FILE="${DEST_DIR}/copy_log.txt"
MISSING_FILE="${DEST_DIR}/missing_connectomes.txt"

# Validate source directory
if [ ! -d "${SOURCE_DIR}" ]; then
    echo -e "${RED}[ERROR]${NC} Source directory does not exist: ${SOURCE_DIR}"
    exit 1
fi

# Create destination directory
mkdir -p "${DEST_DIR}"

# Initialize log files
echo "Connectome Matrix Copy Log - $(date)" > "${LOG_FILE}"
echo "Source Directory: ${SOURCE_DIR}" >> "${LOG_FILE}"
echo "Destination Directory: ${DEST_DIR}" >> "${LOG_FILE}"
echo "================================================" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

echo "Missing Connectome Matrices Report - $(date)" > "${MISSING_FILE}"
echo "Subjects with missing connectome files:" >> "${MISSING_FILE}"
echo "================================================" >> "${MISSING_FILE}"
echo "" >> "${MISSING_FILE}"

# Counters
total_subjects=0
successful_copies=0
subjects_with_missing=0
total_missing_files=0

echo -e "${GREEN}[INFO]${NC} Starting connectome matrix copy process..."
echo -e "${GREEN}[INFO]${NC} Source: ${SOURCE_DIR}"
echo -e "${GREEN}[INFO]${NC} Destination: ${DEST_DIR}"
echo ""

# Get list of all subject directories (6-digit numbers)
subjects=$(ls "${SOURCE_DIR}" | grep -E '^[0-9]{6}$' | sort)

if [ -z "${subjects}" ]; then
    echo -e "${RED}[ERROR]${NC} No subject directories found in ${SOURCE_DIR}"
    exit 1
fi

total_subjects=$(echo "${subjects}" | wc -l)
echo -e "${GREEN}[INFO]${NC} Found ${total_subjects} subjects to process"
echo ""

# Process each subject
current=1
for subject in ${subjects}; do
    echo -e "${GREEN}[INFO]${NC} Processing subject ${subject} (${current}/${total_subjects})"
    
    # Define source paths
    source_subject_dir="${SOURCE_DIR}/${subject}"
    source_output_dir="${source_subject_dir}/output"
    connectome1="${source_output_dir}/connectome_matrix_aparc.a2009s+aseg.csv"
    connectome2="${source_output_dir}/connectome_matrix_aparc+aseg.csv"
    
    # Define destination paths
    dest_subject_dir="${DEST_DIR}/${subject}"
    mkdir -p "${dest_subject_dir}"
    
    # Check if files exist and copy them
    missing_files=()
    copied_files=()
    
    # Check and copy aparc.a2009s+aseg connectome
    if [ -f "${connectome1}" ]; then
        cp "${connectome1}" "${dest_subject_dir}/"
        if [ $? -eq 0 ]; then
            copied_files+=("connectome_matrix_aparc.a2009s+aseg.csv")
            echo -e "  ${GREEN}✓${NC} Copied aparc.a2009s+aseg connectome"
        else
            echo -e "  ${RED}✗${NC} Failed to copy aparc.a2009s+aseg connectome"
            missing_files+=("connectome_matrix_aparc.a2009s+aseg.csv (copy failed)")
        fi
    else
        echo -e "  ${YELLOW}!${NC} Missing aparc.a2009s+aseg connectome"
        missing_files+=("connectome_matrix_aparc.a2009s+aseg.csv")
    fi
    
    # Check and copy aparc+aseg connectome
    if [ -f "${connectome2}" ]; then
        cp "${connectome2}" "${dest_subject_dir}/"
        if [ $? -eq 0 ]; then
            copied_files+=("connectome_matrix_aparc+aseg.csv")
            echo -e "  ${GREEN}✓${NC} Copied aparc+aseg connectome"
        else
            echo -e "  ${RED}✗${NC} Failed to copy aparc+aseg connectome"
            missing_files+=("connectome_matrix_aparc+aseg.csv (copy failed)")
        fi
    else
        echo -e "  ${YELLOW}!${NC} Missing aparc+aseg connectome"
        missing_files+=("connectome_matrix_aparc+aseg.csv")
    fi
    
    # Log results
    echo "Subject: ${subject}" >> "${LOG_FILE}"
    echo "  Source: ${source_output_dir}" >> "${LOG_FILE}"
    echo "  Destination: ${dest_subject_dir}" >> "${LOG_FILE}"
    
    if [ ${#copied_files[@]} -gt 0 ]; then
        echo "  Copied files: ${copied_files[*]}" >> "${LOG_FILE}"
    fi
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "  Missing files: ${missing_files[*]}" >> "${LOG_FILE}"
        echo "${subject}: ${missing_files[*]}" >> "${MISSING_FILE}"
        subjects_with_missing=$((subjects_with_missing + 1))
        total_missing_files=$((total_missing_files + ${#missing_files[@]}))
    fi
    
    if [ ${#copied_files[@]} -gt 0 ]; then
        successful_copies=$((successful_copies + 1))
    fi
    
    echo "" >> "${LOG_FILE}"
    echo ""
    current=$((current + 1))
done

# Summary
echo "================================================" >> "${LOG_FILE}"
echo "SUMMARY:" >> "${LOG_FILE}"
echo "Total subjects processed: ${total_subjects}" >> "${LOG_FILE}"
echo "Subjects with successful copies: ${successful_copies}" >> "${LOG_FILE}"
echo "Subjects with missing files: ${subjects_with_missing}" >> "${LOG_FILE}"
echo "Total missing files: ${total_missing_files}" >> "${LOG_FILE}"
echo "Copy completed: $(date)" >> "${LOG_FILE}"

echo "" >> "${MISSING_FILE}"
echo "================================================" >> "${MISSING_FILE}"
echo "SUMMARY:" >> "${MISSING_FILE}"
echo "Total subjects with missing files: ${subjects_with_missing}" >> "${MISSING_FILE}"
echo "Total missing files: ${total_missing_files}" >> "${MISSING_FILE}"
echo "Report generated: $(date)" >> "${MISSING_FILE}"

# Display final summary
echo -e "${GREEN}[INFO]${NC} ================================================"
echo -e "${GREEN}[INFO]${NC} COPY PROCESS COMPLETED"
echo -e "${GREEN}[INFO]${NC} ================================================"
echo -e "${GREEN}[INFO]${NC} Total subjects processed: ${total_subjects}"
echo -e "${GREEN}[INFO]${NC} Subjects with successful copies: ${successful_copies}"
echo -e "${GREEN}[INFO]${NC} Subjects with missing files: ${subjects_with_missing}"
echo -e "${GREEN}[INFO]${NC} Total missing files: ${total_missing_files}"
echo ""
echo -e "${GREEN}[INFO]${NC} Files created:"
echo -e "${GREEN}[INFO]${NC}   - Log file: ${LOG_FILE}"
echo -e "${GREEN}[INFO]${NC}   - Missing files report: ${MISSING_FILE}"
echo -e "${GREEN}[INFO]${NC}   - Connectome matrices copied to: ${DEST_DIR}"

if [ ${subjects_with_missing} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}[WARNING]${NC} ${subjects_with_missing} subjects have missing connectome files."
    echo -e "${YELLOW}[WARNING]${NC} See ${MISSING_FILE} for details."
fi

echo ""
echo -e "${GREEN}[INFO]${NC} Copy process completed successfully!"