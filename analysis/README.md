# Connectome Analysis Scripts

This directory contains reorganized and refactored connectome analysis scripts with shared metric computation utilities.

## Directory Structure

```
analysis/
├── README.md                           # This file
├── intra_inter_subject_analysis.py    # Intra- and inter-subject comparison analysis
├── population_average_analysis.py     # Population average vs test subjects analysis
├── utils/
│   └── analysis_metrics.py            # Shared metric computation functions
├── test_results/                      # Output from intra/inter-subject analysis
│   ├── all_subject_results.json
│   ├── analysis_summary.json
│   └── plots/
└── population_analysis/               # Output from population average analysis
    ├── population_average_*.npy
    ├── test_vs_population_results.csv
    └── *.png (visualizations)
```

## Analysis Scripts

### 1. Intra- and Inter-Subject Analysis (`intra_inter_subject_analysis.py`)

**Purpose**: Compares predicted connectomes against ground truth using two approaches:
- **Intra-subject**: Predicted vs true for the same subject (model accuracy)
- **Inter-subject**: Predicted vs true from other subjects (subject specificity)

**Key Features**:
- Processes all connectome types: NOS, FA, MD, AD, RD, SIFT2
- Handles both atlases: aparc+aseg (84 ROIs), aparc.a2009s+aseg (164 ROIs)
- Zero masking for diffusion metrics (FA, MD, AD, RD)
- Result caching with JSON for efficient re-analysis
- Violin plots with statistical significance testing

**Usage**:
```bash
# Full analysis
python analysis/intra_inter_subject_analysis.py \
    --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt

# Load existing results and regenerate plots only
python analysis/intra_inter_subject_analysis.py \
    --subject-list /media/volume/MV_HCP/subjects_tractography_output_1000_test.txt \
    --load-only --plot-only
```

**Metrics Computed**:
- Pearson correlation (k=0, includes diagonal)
- LERM (Log-Euclidean Riemannian Metric) using matrix logarithm
- Connection counts
- Mean values

### 2. Population Average Analysis (`population_average_analysis.py`)

**Purpose**: Compares test subject predictions against population-weighted average connectomes from training set.

**Key Features**:
- Computes population average from 690 training subjects
- Compares 200 test subjects to population average
- Uses same metrics as intra/inter analysis for consistency
- Caches population averages for efficiency
- Comprehensive visualization and reporting

**Usage**:
```bash
# Full analysis (compute averages + test comparisons)
python analysis/population_average_analysis.py --full-analysis

# Only compute population averages
python analysis/population_average_analysis.py --compute-average

# Only run test comparisons (requires existing averages)
python analysis/population_average_analysis.py --test-comparison
```

**Metrics Computed**:
- Pearson correlation (k=0, includes diagonal)
- LERM using matrix logarithm
- RMSE and MAE (for backwards compatibility)
- Connection counts

## Shared Utilities

### `utils/analysis_metrics.py`

This module contains shared metric computation functions used by both analysis scripts to ensure consistency.

**Functions**:

1. **`apply_zero_mask(true_matrix, pred_matrix, connectome_type, mask_zeros, logger)`**
   - Applies zero masking for diffusion metrics (FA, MD, AD, RD)
   - Masks entries that are zero in either true or pred matrix
   - Returns: masked_true, masked_pred, mask

2. **`compute_correlation(true_matrix, pred_matrix, include_diagonal=True)`**
   - Computes Pearson correlation
   - Uses k=0 to include diagonal (validated approach)
   - Filters out zeros for masked matrices
   - Returns: correlation coefficient

3. **`compute_lerm(true_matrix, pred_matrix, use_matrix_log=True, epsilon=1e-10)`**
   - Computes Log-Euclidean Riemannian Metric
   - Uses scipy.linalg.logm() for matrix logarithm (NOT element-wise log)
   - Adds epsilon to avoid log(0)
   - Returns: LERM distance (Frobenius norm)

4. **`compute_connectome_metrics(true_matrix, pred_matrix, connectome_type, mask_zeros, logger)`**
   - Comprehensive wrapper that computes all metrics
   - Returns: dict with correlation, lerm, connection counts, means, etc.

## Important Implementation Details

### Metric Computation

**Critical**: Both analysis scripts use identical metric formulas from `analysis_metrics.py`:

1. **Pearson Correlation**:
   ```python
   # k=0 includes diagonal (validated approach)
   triu_indices = np.triu_indices_from(matrix, k=0)
   correlation = np.corrcoef(true[triu_indices], pred[triu_indices])[0, 1]
   ```

2. **LERM (Log-Euclidean Riemannian Metric)**:
   ```python
   # Use matrix logarithm, NOT element-wise log
   from scipy.linalg import logm, norm
   lerm = norm(logm(pred) - logm(true), 'fro')
   ```
   
   **Warning**: Element-wise `np.log()` gives vastly different results (~100x larger) than matrix logarithm `logm()`. Always use `logm()`.

3. **Zero Masking**:
   - Only applied to diffusion metrics: FA, MD, AD, RD
   - Not applied to NOS or SIFT2
   - Masks entries where either true OR pred is zero
   ```python
   mask = (true != 0) & (pred != 0)
   ```
   - Improves FA correlation by ~27% (0.756 → 0.966 in test case)

### Why This Organization?

**Before**: 
- `improved_multi_subject_analysis.py` - correct metrics with logm()
- `population_connectome_analysis.py` - incorrect metrics with np.log()
- No code sharing, inconsistent results

**After**:
- `intra_inter_subject_analysis.py` - uses shared metrics
- `population_average_analysis.py` - uses shared metrics
- `utils/analysis_metrics.py` - single source of truth
- Guaranteed identical metric computation

This ensures:
✅ Apples-to-apples comparison between intra/inter and population analyses
✅ Maintainability - fix bugs in one place
✅ Consistency - same formulas everywhere
✅ Correctness - validated metric implementations

## Migration Notes

If you have existing code referencing old script names:
- `improved_multi_subject_analysis.py` → `analysis/intra_inter_subject_analysis.py`
- `population_connectome_analysis.py` → `analysis/population_average_analysis.py`

Existing cached results remain compatible, but population analysis results computed before reorganization may differ due to corrected LERM calculation.

## Future Enhancements

Potential improvements:
1. Extend population analysis to support all connectome types (FA, MD, AD, RD, SIFT2)
2. Add combined population vs intra/inter comparison plots
3. Support for multiple population stratifications (age, sex, etc.)
4. Parallel processing for faster population average computation
