#!/usr/bin/env python3
"""
Migration Guide and Compatibility Script

This script demonstrates how to migrate from the old metrics_connectome.py 
to the new unified_connectome.py system, and provides compatibility functions.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from current directory when run as script
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unified_connectome import ConnectomeAnalyzer, analyze_connectomes_from_labels
    from connectome_config import ATLAS_CONFIG
else:
    from utils.unified_connectome import ConnectomeAnalyzer, analyze_connectomes_from_labels
    from utils.connectome_config import ATLAS_CONFIG
import warnings


class LegacyConnectomeMetrics:
    """
    Legacy compatibility wrapper that mimics the old ConnectomeMetrics interface
    while using the new unified system under the hood.
    
    This allows old code to continue working with minimal changes.
    """
    
    def __init__(self, true_labels=None, pred_labels=None, encoding='symmetric', 
                 atlas="aparc+aseg", out_path='output', graph=False, plot=True):
        
        warnings.warn(
            "LegacyConnectomeMetrics is deprecated. Please use ConnectomeAnalyzer "
            "from utils.unified_connectome for new code.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Initialize the new analyzer
        self.analyzer = ConnectomeAnalyzer(atlas=atlas, out_path=out_path)
        self.atlas = atlas
        self.encoding = encoding
        self.out_path = out_path
        
        # Store for compatibility
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.num_labels = ATLAS_CONFIG[atlas]['num_labels']
        
        if true_labels is not None and pred_labels is not None:
            # Create connectomes using the new system
            self.analyzer.create_connectome_from_labels(
                pred_labels, encoding=encoding, connectome_name='pred'
            )
            self.analyzer.create_connectome_from_labels(
                true_labels, encoding=encoding, connectome_name='true'
            )
            
            # Store for compatibility
            self.pred_connectome = self.analyzer.connectomes['pred']
            self.true_connectome = self.analyzer.connectomes['true']
            
            # Compute metrics
            self.analyzer.compute_comparison_metrics('true', 'pred', 'comparison')
            self.results = self.analyzer.metrics['comparison']
            
            # Save connectomes (legacy format)
            self.analyzer.save_connectome('true', f'connectome_{atlas}_true.csv')
            self.analyzer.save_connectome('pred', f'connectome_{atlas}_pred.csv')
            
            if plot:
                self.analyzer.create_comparison_plot('true', 'pred', f'{atlas}_comparison')
            
            if graph:
                self.analyzer.compute_network_metrics('true')
                self.analyzer.compute_network_metrics('pred')
                # Merge network metrics into results for compatibility
                self.results.update({
                    f'{k}_true': v for k, v in self.analyzer.network_metrics['true'].items()
                })
                self.results.update({
                    f'{k}_pred': v for k, v in self.analyzer.network_metrics['pred'].items()
                })
    
    def format_metrics(self):
        """Legacy format_metrics method for compatibility"""
        if not hasattr(self, 'results') or not self.results:
            return "No metrics computed."
        
        # Map new metric names to old ones for compatibility
        metric_mapping = {
            'pearson_r': 'Pearson Correlation',
            'spearman_r': 'Spearman Correlation',
            'mse': 'MSE',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'frobenius_norm': 'Frobenius Norm',
            'wasserstein_distance': 'Earth Mover\'s Distance'
        }
        
        formatted_results = {}
        for new_key, old_key in metric_mapping.items():
            if new_key in self.results:
                formatted_results[old_key] = self.results[new_key]
        
        return f"""
        Metrics Summary (Legacy Format):
        --------------------------------
        Pearson Correlation: {formatted_results.get('Pearson Correlation', 'N/A'):.4f}
        Spearman Correlation: {formatted_results.get('Spearman Correlation', 'N/A'):.4f}
        MSE: {formatted_results.get('MSE', 'N/A'):.4f}
        RMSE: {formatted_results.get('RMSE', 'N/A'):.4f}
        MAE: {formatted_results.get('MAE', 'N/A'):.4f}
        Frobenius Norm: {formatted_results.get('Frobenius Norm', 'N/A'):.4f}
        Earth Mover's Distance: {formatted_results.get("Earth Mover's Distance", 'N/A'):.4f}
        
        Note: This is a legacy compatibility wrapper. 
        Please use ConnectomeAnalyzer for new code.
        """


def migrate_old_code_example():
    """
    Example showing how to migrate from old code to new code
    """
    print("=== MIGRATION EXAMPLE ===\n")
    
    # OLD WAY (deprecated, but still works via compatibility wrapper)
    print("OLD WAY (using compatibility wrapper):")
    print("=" * 40)
    print("""
    # Old import (still works but deprecated)
    from utils.metrics_connectome import ConnectomeMetrics
    
    # Old usage
    CM = ConnectomeMetrics(
        true_labels=true_labels, 
        pred_labels=pred_labels, 
        atlas=atlas, 
        out_path=output_path, 
        graph=True
    )
    print(CM.format_metrics())
    """)
    
    # NEW WAY (recommended)
    print("\nNEW WAY (recommended):")
    print("=" * 40)
    print("""
    # New import
    from utils.unified_connectome import ConnectomeAnalyzer, analyze_connectomes_from_labels
    
    # Method 1: High-level function (easiest)
    analyzer = analyze_connectomes_from_labels(
        pred_labels_file='predictions_symmetric.txt',
        true_labels_file='labels_symmetric.txt',
        diffusion_metrics_dir='path/to/diffusion/metrics',
        atlas='aparc+aseg',
        out_path='output'
    )
    
    # Method 2: Step-by-step control
    analyzer = ConnectomeAnalyzer(atlas='aparc+aseg', out_path='output')
    analyzer.create_connectome_from_labels(pred_labels, connectome_name='pred')
    analyzer.create_connectome_from_labels(true_labels, connectome_name='true')
    analyzer.compute_comparison_metrics('true', 'pred')
    analyzer.create_comparison_plot('true', 'pred')
    analyzer.print_summary()
    """)


def feature_comparison():
    """
    Compare features between old and new systems
    """
    print("\n=== FEATURE COMPARISON ===\n")
    
    features = [
        ("Connectome Creation", "✓ Basic", "✓ Advanced (weighted, multiple types)"),
        ("Visualization", "✓ Basic plots", "✓ Comprehensive comparison plots"),
        ("Statistical Metrics", "✓ Basic", "✓ Comprehensive (15+ metrics)"),
        ("Network Analysis", "✓ Limited", "✓ Extensive (10+ network metrics)"),
        ("Error Handling", "❌ Poor", "✓ Robust"),
        ("Code Quality", "❌ Messy", "✓ Clean, documented"),
        ("Extensibility", "❌ Difficult", "✓ Easy to extend"),
        ("Performance", "❌ Slow", "✓ Optimized"),
        ("Multiple Atlases", "❌ Limited", "✓ Full support"),
        ("Diffusion Metrics", "❌ Not integrated", "✓ Full integration"),
        ("Documentation", "❌ Poor", "✓ Comprehensive"),
        ("Testing", "❌ None", "✓ Built-in validation"),
    ]
    
    print(f"{'Feature':<20} {'Old System':<25} {'New System':<30}")
    print("-" * 75)
    for feature, old, new in features:
        print(f"{feature:<20} {old:<25} {new:<30}")


def performance_improvements():
    """
    Highlight performance improvements in the new system
    """
    print("\n=== PERFORMANCE IMPROVEMENTS ===\n")
    
    improvements = [
        "Vectorized operations instead of loops",
        "Efficient memory usage for large connectomes", 
        "Cached computations to avoid redundancy",
        "Optimized statistical calculations",
        "Parallel-friendly design",
        "Reduced I/O operations",
        "Better error handling (no crashes)",
        "Automatic cleanup of temporary variables"
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement}")


def best_practices():
    """
    Best practices for using the new system
    """
    print("\n=== BEST PRACTICES ===\n")
    
    practices = [
        "Use analyze_connectomes_from_labels() for complete analysis",
        "Use ConnectomeAnalyzer() for custom workflows",
        "Always specify atlas explicitly",
        "Use meaningful connectome names",
        "Save intermediate results with save_connectome()",
        "Check logs for warnings and errors",
        "Use print_summary() to verify results",
        "Store results with save_results_summary()",
        "Handle exceptions in your code",
        "Use the configuration system for new metrics"
    ]
    
    for i, practice in enumerate(practices, 1):
        print(f"{i}. {practice}")


def main():
    """Main migration guide"""
    print("CONNECTOME SYSTEM MIGRATION GUIDE")
    print("=" * 50)
    
    migrate_old_code_example()
    feature_comparison()
    performance_improvements()
    best_practices()
    
    print("\n=== NEXT STEPS ===\n")
    print("1. Update your imports to use unified_connectome")
    print("2. Replace ConnectomeMetrics with ConnectomeAnalyzer")
    print("3. Use the high-level analyze_connectomes_from_labels() function")
    print("4. Test your code with the new system")
    print("5. Remove old metrics_connectome.py imports")
    print("6. Enjoy cleaner, faster, more robust connectome analysis!")
    
    print(f"\n{'='*50}")
    print("For questions or issues, check the documentation in unified_connectome.py")


if __name__ == "__main__":
    main()