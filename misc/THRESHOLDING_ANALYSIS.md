# Connectome Thresholding Analysis Documentation

## Overview

I've added comprehensive thresholding analysis to the unified connectome system. This allows you to analyze connectomes with and without thresholding to remove weak connections and focus on the most robust connectivity patterns.

**Key Feature**: The thresholding is based on **Number of Streamlines (NOS)** connectomes, ensuring that all connectome types (FA, MD, AD, RD) are thresholded using the same anatomical criteria.

## Features Added

### 1. NOS-Based Thresholding (Primary Method)

The system uses **Number of Streamlines (NOS)** connectomes to determine which connections to threshold:

1. **Threshold Calculation**: Computes threshold values based on NOS connectome statistics
2. **Mask Creation**: Creates binary masks based on NOS thresholds
3. **Universal Application**: Applies the same masks to all connectome types (FA, MD, AD, RD)
4. **Anatomical Basis**: Ensures thresholding is based on actual streamline counts, not diffusion values

### 2. Statistical Thresholding Methods

The system supports multiple statistical approaches to determine NOS-based thresholds:

- **Percentile-based**: Remove connections with the lowest X% of streamlines
- **IQR Outliers**: Remove connections with streamline counts below Q1 - 1.5*IQR
- **Standard Deviation**: Remove connections below mean - X*std streamline counts
- **Absolute**: Remove connections below an absolute streamline count threshold

### 3. Automatic Threshold Analysis

The system automatically:
- Detects NOS connectomes for threshold calculation
- Computes threshold statistics for streamline counts
- Applies unified masks to all connectome types
- Compares original vs thresholded performance
- Generates thresholded comparison plots
- Saves detailed threshold analysis results

## Scientific Rationale

### Why NOS-Based Thresholding?

1. **Anatomical Relevance**: Streamline count directly reflects the evidence for anatomical connectivity
2. **Consistent Criteria**: All connectome types use the same anatomical threshold
3. **Noise Reduction**: Removes connections with insufficient streamline support
4. **Interpretability**: Easy to understand "minimum streamlines" threshold
5. **Biological Validity**: Weak streamline bundles may not represent true white matter tracts

### Advantages Over Value-Based Thresholding

- **FA/MD/AD/RD values** can vary widely due to tissue properties
- **Streamline counts** provide consistent evidence of connectivity strength
- **Unified thresholding** ensures fair comparison across all metrics
- **Anatomical grounding** bases decisions on tractography results

## Usage

### In Unified Connectome System

```python
from utils.unified_connectome import ConnectomeAnalyzer

# Create analyzer
analyzer = ConnectomeAnalyzer(atlas="aparc+aseg", out_path="output")

# Add your connectomes (must include NOS connectomes)
analyzer.connectomes['true_nos'] = true_nos_matrix     # Required for thresholding
analyzer.connectomes['pred_nos'] = pred_nos_matrix     # Required for thresholding
analyzer.connectomes['true_fa'] = true_fa_matrix
analyzer.connectomes['pred_fa'] = pred_fa_matrix

# Define threshold methods (applied to NOS connectomes)
threshold_methods = [
    {'method': 'percentile', 'value': 5.0, 'name': 'bottom_5pct'},
    {'method': 'percentile', 'value': 10.0, 'name': 'bottom_10pct'},
    {'method': 'outlier_iqr', 'value': 1.5, 'name': 'iqr_outliers'},
    {'method': 'std', 'value': 1.0, 'name': 'mean_minus_1std'}
]

# Run NOS-based thresholded analysis
connectome_pairs = [('true_nos', 'pred_nos'), ('true_fa', 'pred_fa')]
results = analyzer.create_thresholded_analysis(
    connectome_pairs,
    threshold_methods=threshold_methods
)
```

### Automatic Behavior

1. **NOS Detection**: System automatically finds NOS connectomes (names containing 'nos')
2. **Threshold Calculation**: Computes thresholds based on NOS statistics
3. **Mask Application**: Applies NOS-based masks to all connectome pairs
4. **Results Storage**: Saves threshold masks and comparative results

### Fallback Mechanism

If NOS connectomes are not available, the system falls back to individual thresholding (original behavior).

## Output Files

The NOS-based thresholding analysis creates:

1. **Thresholded connectome CSV files**: `connectome_[name]_thresh_[method]_[atlas].csv`
2. **Thresholded comparison plots**: `thresh_[method]_[comparison]_[atlas].png`
3. **Threshold analysis JSON**: `threshold_analysis_[atlas].json` (includes threshold masks)
4. **Regular analysis results**: All standard metrics and plots

### New in JSON Output

```json
{
  "threshold_masks": {
    "bottom_5pct": {
      "true_threshold": 45.2,
      "pred_threshold": 42.8,
      "true_mask": [[true, false, ...], ...],
      "pred_mask": [[true, false, ...], ...]
    }
  }
}
```

## Threshold Methods Explained

### 1. Percentile Method (`'percentile'`) - **Recommended**
- **Purpose**: Remove connections with the fewest X% of streamlines
- **Value**: Percentage (e.g., 5.0 = remove bottom 5% by streamline count)
- **Example**: Remove connections with < 20 streamlines (if 20 is 5th percentile)
- **Use case**: Remove connections with minimal streamline evidence

### 2. IQR Outliers (`'outlier_iqr'`)
- **Purpose**: Remove connections with unusually low streamline counts
- **Value**: IQR multiplier (typically 1.5)
- **Formula**: Threshold = Q1 - value * IQR of streamline counts
- **Use case**: Conservative removal of statistical outliers only

### 3. Standard Deviation (`'std'`)
- **Purpose**: Remove connections below mean - X standard deviations of streamline counts
- **Value**: Number of standard deviations (e.g., 1.0)
- **Formula**: Threshold = mean_streamlines - value * std_streamlines
- **Use case**: Remove connections significantly below average streamline count

### 4. Absolute Threshold (`'absolute'`)
- **Purpose**: Remove connections below a specific streamline count
- **Value**: Minimum number of streamlines
- **Example**: Remove all connections with < 10 streamlines
- **Use case**: Domain-specific minimum streamline requirements

## Example Results

From testing, NOS-based thresholding typically:
- **Bottom 5%**: Removes connections with very few streamlines, improves signal-to-noise
- **Bottom 10%**: More aggressive removal of weak connections
- **IQR Outliers**: Conservative, removes only statistical outliers in streamline counts
- **Absolute (e.g., 10)**: Removes all connections below minimum streamline threshold

## Benefits of NOS-Based Approach

### Scientific Benefits
- **Anatomical Consistency**: All metrics use same anatomical threshold
- **Biological Relevance**: Based on actual white matter evidence
- **Noise Reduction**: Removes spurious low-streamline connections
- **Interpretability**: Clear "minimum streamlines" criterion

### Analysis Benefits
- **Fair Comparison**: All connectome types thresholded consistently
- **Preserved Relationships**: Maintains relative diffusion metric relationships
- **Unified Masks**: Same connections removed across all metrics
- **Clear Rationale**: Easy to explain and justify thresholding decisions

## Configuration

### Default NOS-Based Methods
```python
default_methods = [
    {'method': 'percentile', 'value': 5.0, 'name': 'bottom_5pct'},
    {'method': 'percentile', 'value': 10.0, 'name': 'bottom_10pct'},
    {'method': 'outlier_iqr', 'value': 1.5, 'name': 'iqr_outliers'},
    {'method': 'std', 'value': 1.0, 'name': 'mean_minus_1std'}
]
```

### Recommended Settings
- **Conservative**: `bottom_5pct` only
- **Moderate**: `bottom_5pct` and `bottom_10pct`
- **Aggressive**: Include `mean_minus_1std`
- **Custom**: Set absolute streamline thresholds based on your data

## Integration with Existing Workflow

The NOS-based thresholding analysis is fully integrated:
- ✅ **Automatic NOS Detection**: Finds NOS connectomes automatically
- ✅ **Universal Application**: Works with all diffusion metrics
- ✅ **Atlas Compatibility**: Works with both atlases (aparc+aseg, aparc.a2009s+aseg)
- ✅ **Network Analysis**: Includes network analysis on thresholded connectomes
- ✅ **Backward Compatibility**: Falls back to individual thresholding if needed
- ✅ **Optional Feature**: Can be disabled if not needed (`include_thresholding=False`)