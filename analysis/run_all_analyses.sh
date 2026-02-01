#!/bin/bash

#python compute_trt_similarity.py --subjects_file /media/volume/MV_HCP/subjects_tractography_output_TRT.txt --test_path /media/volume/MV_HCP/HCP_MRtrix_test --retest_path /media/volume/MV_HCP/HCP_MRtrix_retest

################################################################################
# Run All Intra/Inter-Subject Connectome Analyses
#
# This script runs three variants of the analysis in sequence:
# 1. Normal analysis (with diagonal included)
# 2. No-diagonal analysis (diagonal excluded from metrics)
# 3. Length-filtered analysis (streamlines >= 20mm)
#
# Each analysis generates its own results directory with plots and statistics.
#
# Usage:
#   ./run_all_analyses.sh <subject_list_file> [num_processes]
#
# Example:
#   ./run_all_analyses.sh /path/to/subjects.txt 8
# bash /media/volume/HCP_diffusion_MV/DeepMultiConnectome/analysis/run_all_analyses.sh /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt

################################################################################

# bash /media/volume/HCP_diffusion_MV/DeepMultiConnectome/tractography/tractography.sh /media/volume/MV_HCP/subjects_tractography_output_1000_train_200.txt /media/volume/MV_HCP/HCP_MRtrix 4 16

#! include prediction for tractoinferno

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <subject_list_file> [num_processes]"
    echo "Example: $0 /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt 8"
    exit 1
fi

SUBJECT_LIST=$1
NUM_PROCESSES=${2:-16}  # Default to 16 processes if not specified

# Check if subject list exists
if [ ! -f "$SUBJECT_LIST" ]; then
    echo "Error: Subject list file not found: $SUBJECT_LIST"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ANALYSIS_SCRIPT="$SCRIPT_DIR/intra_inter_subject_analysis.py"

# Check if analysis script exists
if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script not found: $ANALYSIS_SCRIPT"
    exit 1
fi

# Log file
LOG_DIR="$SCRIPT_DIR/analysis_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_all_analyses_${TIMESTAMP}.log"

echo "================================================================================"
echo "Running All Connectome Analyses"
echo "================================================================================"
echo "Subject list: $SUBJECT_LIST"
echo "Num processes: $NUM_PROCESSES"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Function to run analysis and log output
run_analysis() {
    local description=$1
    local cmd=$2
    
    echo "--------------------------------------------------------------------------------"
    echo "[$description]"
    echo "Command: $cmd"
    echo "Start: $(date)"
    echo "--------------------------------------------------------------------------------"
    
    # Run command and tee to both console and log file
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        echo ""
        echo "✓ $description completed successfully"
        echo "End: $(date)"
        echo ""
    else
        echo ""
        echo "✗ $description failed!"
        echo "End: $(date)"
        echo "Check log file for details: $LOG_FILE"
        echo ""
        return 1
    fi
}

# Start logging
{
    echo "================================================================================"
    echo "All Connectome Analyses - Full Log"
    echo "Start time: $(date)"
    echo "Subject list: $SUBJECT_LIST"
    echo "Num processes: $NUM_PROCESSES"
    echo "================================================================================"
    echo ""
} > "$LOG_FILE"

# Analysis 1: Normal (with diagonal)
run_analysis \
    "Analysis 1: Normal (with diagonal)" \
    "python3 '$ANALYSIS_SCRIPT' --subject-list '$SUBJECT_LIST' --processes $NUM_PROCESSES"

# Analysis 2: No-diagonal (diagonal excluded)
run_analysis \
    "Analysis 2: No-diagonal (diagonal excluded from metrics)" \
    "python3 '$ANALYSIS_SCRIPT' --subject-list '$SUBJECT_LIST' --processes $NUM_PROCESSES --no-diagonal"

# Analysis 3: Length-filtered (streamlines >= 20mm)
run_analysis \
    "Analysis 3: Length-filtered (streamlines >= 20mm)" \
    "python3 '$ANALYSIS_SCRIPT' --subject-list '$SUBJECT_LIST' --processes $NUM_PROCESSES --min-length-mm 20"

# Final summary
echo "================================================================================"
echo "All Analyses Completed Successfully!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  1. Normal: $SCRIPT_DIR/interintra_similarity_results/"
echo "  2. No-diagonal: $SCRIPT_DIR/interintra_similarity_results_nodiagonal/"
echo "  3. Length-filtered: $SCRIPT_DIR/interintra_similarity_results_20mm/"
echo ""
echo "Full log: $LOG_FILE"
echo "================================================================================"

# Append final summary to log
{
    echo ""
    echo "================================================================================"
    echo "All analyses completed successfully!"
    echo "End time: $(date)"
    echo "================================================================================"
} >> "$LOG_FILE"
