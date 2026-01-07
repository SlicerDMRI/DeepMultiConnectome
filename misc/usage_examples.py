#!/usr/bin/env python3
"""
Usage Examples for Improved Multi-Subject Connectome Analysis

This script demonstrates how to use the improved multi-subject analysis with result caching
and flexible processing modes.
"""

import os
from pathlib import Path

def main():
    print("="*80)
    print("IMPROVED MULTI-SUBJECT CONNECTOME ANALYSIS - USAGE EXAMPLES")
    print("="*80)
    
    print("""
    The improved multi-subject analysis provides the following key features:
    
    1. ✅ Connectomes saved in HCP_MRtrix directory structure (like complete_subject_analysis.py)
    2. ✅ JSON-based result caching for efficient re-analysis  
    3. ✅ Flexible processing modes (full analysis, load-only, plot-only)
    4. ✅ Parallel processing support
    5. ✅ Inter and intra-subject correlation & LERM metrics
    6. ✅ All 6 connectome types (NoS, FA, MD, AD, RD, SIFT2)
    7. ✅ Results saved in centralized location for easy access
    
    """)
    
    print("USAGE EXAMPLES:")
    print("-" * 50)
    
    print("\n1. FIRST TIME - Full Analysis (with parallel processing):")
    print("   python3 improved_multi_subject_analysis.py \\")
    print("     --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt \\")
    print("     --processes 4")
    
    print("\n2. LIMITED SUBJECTS - Testing with subset:")
    print("   python3 improved_multi_subject_analysis.py \\")
    print("     --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt \\")
    print("     --max-subjects 10 \\")
    print("     --processes 2")
    
    print("\n3. LOAD EXISTING RESULTS - Skip computation, just load and plot:")
    print("   python3 improved_multi_subject_analysis.py \\")
    print("     --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt \\")
    print("     --load-only")
    
    print("\n4. PLOT ONLY - Create new plots from existing results:")
    print("   python3 improved_multi_subject_analysis.py \\")
    print("     --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt \\")
    print("     --plot-only")
    
    print("\n5. FORCE RECOMPUTE - Force recomputation even if results exist:")
    print("   python3 improved_multi_subject_analysis.py \\")
    print("     --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt \\")
    print("     --force-recompute \\")
    print("     --processes 4")
    
    print("\n6. SEQUENTIAL PROCESSING - No parallel processing:")
    print("   python3 improved_multi_subject_analysis.py \\")
    print("     --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt \\")
    print("     --processes 1")
    
    print("\n" + "="*80)
    print("FILE LOCATIONS:")
    print("="*80)
    
    print("\n📁 CONNECTOME CSV FILES (per subject):")
    print("   Location: /media/volume/MV_HCP/HCP_MRtrix/{subject_id}/analysis/{atlas}/")
    print("   Format: connectome_{condition}_{type}_{atlas}.csv")
    print("   Example: /media/volume/MV_HCP/HCP_MRtrix/100206/analysis/aparc+aseg/connectome_true_nos_aparc+aseg.csv")
    
    print("\n📊 CENTRALIZED RESULTS:")
    print("   Location: /media/volume/HCP_diffusion_MV/DeepMultiConnectome/analysis/test_results/")
    print("   Files:")
    print("   - all_subject_results.json      (complete results for all subjects)")
    print("   - analysis_summary.json         (summary statistics across subjects)")
    print("   - all_metrics_combined.csv      (flattened CSV for easy analysis)")
    print("   - plots/                        (violin plots for each connectome type)")
    
    print("\n🔄 RESULT CACHING:")
    print("   - Results are automatically saved to JSON after computation")
    print("   - Use --load-only to skip computation and use existing results")
    print("   - Use --plot-only to create new plots from existing results")
    print("   - Use --force-recompute to override existing results")
    
    print("\n📈 METRICS COMPUTED:")
    print("   - Intra-subject correlation (predicted vs true for same subject)")
    print("   - Inter-subject correlation (predicted vs other subjects' true)")
    print("   - LERM (Log-Euclidean Riemannian Metric) distances")
    print("   - Connection counts and strengths")
    print("   - Statistical comparisons with p-values")
    
    print("\n" + "="*80)
    print("JSON STRUCTURE EXAMPLE:")
    print("="*80)
    
    example_json = """
    {
      "metadata": {
        "subjects": ["100206", "101006", "102109"],
        "atlases": ["aparc+aseg", "aparc.a2009s+aseg"],
        "connectome_types": ["nos", "fa", "md", "ad", "rd", "sift2"],
        "total_subjects": 3,
        "computation_time": 0.92,
        "timestamp": "2025-10-02 10:16:22"
      },
      "results": {
        "100206": {
          "aparc+aseg": {
            "nos": {
              "intra_r": 0.9696,
              "intra_lerm": 534.13,
              "inter_r": 0.9077,
              "inter_lerm": 707.59,
              "true_connections": 4464,
              "pred_connections": 4058
            }
          }
        }
      }
    }
    """
    print(example_json)
    
    print("\n" + "="*80)
    print("PERFORMANCE NOTES:")
    print("="*80)
    
    print("""
    ⚡ PARALLEL PROCESSING:
    - Default: uses half of available CPU cores
    - Specify --processes N for custom number
    - Use --processes 1 for sequential processing
    
    💾 MEMORY EFFICIENCY:
    - Connectomes processed and saved immediately
    - Memory freed after each subject
    - Inter-subject metrics computed efficiently
    
    🚀 SMART CACHING:
    - Connectome CSV files are cached (skipped if exist)
    - Full results cached in JSON format
    - Only recompute what's needed
    
    📊 SCALABILITY:
    - Tested with 10+ subjects successfully
    - Can handle full HCP dataset
    - Results accumulate progressively
    """)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()