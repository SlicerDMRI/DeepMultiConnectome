"""
Configuration file for connectome analysis

This file defines the different types of connectome analyses that can be performed
and the metrics that should be computed for each comparison.
"""

# Atlas configurations
ATLAS_CONFIG = {
    'aparc+aseg': {
        'num_labels': 85,
        'description': 'Desikan-Killiany Atlas with subcortical structures'
    },
    'aparc.a2009s+aseg': {
        'num_labels': 165,
        'description': 'Destrieux Atlas (aparc.a2009s) with subcortical structures'
    }
}

# Diffusion metrics configuration
DIFFUSION_METRICS = {
    'fa': {
        'filename': 'mean_fa_per_streamline.txt',
        'description': 'Fractional Anisotropy',
        'units': 'unitless',
        'range': [0, 1]
    },
    'md': {
        'filename': 'mean_md_per_streamline.txt',
        'description': 'Mean Diffusivity',
        'units': 'mm²/s',
        'range': [0, 0.003]
    },
    'ad': {
        'filename': 'mean_ad_per_streamline.txt',
        'description': 'Axial Diffusivity',
        'units': 'mm²/s',
        'range': [0, 0.005]
    },
    'rd': {
        'filename': 'mean_rd_per_streamline.txt',
        'description': 'Radial Diffusivity',
        'units': 'mm²/s',
        'range': [0, 0.003]
    }
}

# Connectome types
CONNECTOME_TYPES = {
    'nos': {
        'description': 'Number of Streamlines',
        'weight_file': None,
        'units': 'count'
    },
    'fa': {
        'description': 'FA-weighted',
        'weight_file': 'mean_fa_per_streamline.txt',
        'units': 'mean FA'
    },
    'md': {
        'description': 'MD-weighted',
        'weight_file': 'mean_md_per_streamline.txt',
        'units': 'mean MD (mm²/s)'
    },
    'ad': {
        'description': 'AD-weighted',
        'weight_file': 'mean_ad_per_streamline.txt',
        'units': 'mean AD (mm²/s)'
    },
    'rd': {
        'description': 'RD-weighted',
        'weight_file': 'mean_rd_per_streamline.txt',
        'units': 'mean RD (mm²/s)'
    }
}

# Comparison metrics to compute
COMPARISON_METRICS = [
    'pearson_r',      # Pearson correlation coefficient
    'pearson_p',      # Pearson correlation p-value
    'spearman_r',     # Spearman correlation coefficient
    'spearman_p',     # Spearman correlation p-value
    'mean_lerm',      # Linear Error in Relative Magnitude
    'MAE',            # Mean Absolute Error
    'MSE',            # Mean Squared Error
    'RMSE'            # Root Mean Squared Error
]

# Output file patterns
OUTPUT_PATTERNS = {
    'connectome_pred': 'connectome_pred_{metric}_{atlas}.csv',
    'connectome_true': 'connectome_true_{metric}_{atlas}.csv',
    'comparison_plot': 'connectome_comparison_{metric}_{atlas}.png',
    'comparison_summary': 'connectome_comparison_summary_{atlas}.csv',
    'predictions_decoded': 'predictions_{atlas}.txt',
    'predictions_symmetric': 'predictions_{atlas}_symmetric.txt'
}

# Default file locations relative to subject directory
DEFAULT_PATHS = {
    'tractography_file': 'output/streamlines_10M.vtk',
    'true_labels_raw': 'output/labels_10M_{atlas}.txt',
    'true_labels_symmetric': 'output/labels_10M_{atlas}_symmetric.txt',
    'diffusion_metrics_dir': 'dMRI',
    'output_dir': 'output'
}

# Visualization settings
PLOT_CONFIG = {
    'figsize': (15, 12),
    'dpi': 300,
    'colormap': 'viridis',
    'scatter_alpha': 0.5,
    'scatter_size': 1,
    'difference_colormap': 'RdBu_r'
}

# Analysis thresholds and settings
ANALYSIS_CONFIG = {
    'min_connection_threshold': 0,  # Minimum connection strength to include in analysis
    'exclude_diagonal': True,       # Exclude self-connections from correlation analysis
    'symmetric_encoding': True,     # Use symmetric encoding for labels
    'log_transform': False          # Apply log transform to connectome values (for visualization)
}