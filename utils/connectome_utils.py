import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import create_logger

class ConnectomeBuilder:
    """
    Class to build different types of connectomes from streamline labels and diffusion metrics
    """
    
    def __init__(self, num_labels, out_path, logger=None):
        self.num_labels = num_labels
        self.out_path = out_path
        self.logger = logger if logger else self._create_default_logger()
        
    def _create_default_logger(self):
        """Create a default logger if none provided"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def load_streamline_labels(self, labels_file):
        """Load streamline labels from file"""
        try:
            with open(labels_file, 'r') as f:
                labels = [int(line.strip()) for line in f if line.strip()]
            self.logger.info(f"Loaded {len(labels)} streamline labels from {labels_file}")
            return np.array(labels)
        except Exception as e:
            self.logger.error(f"Error loading labels from {labels_file}: {e}")
            return None
    
    def load_diffusion_metrics(self, metrics_file):
        """Load diffusion metrics (FA, MD, AD, RD) from file"""
        try:
            with open(metrics_file, 'r') as f:
                metrics = [float(line.strip()) for line in f if line.strip()]
            self.logger.info(f"Loaded {len(metrics)} diffusion metric values from {metrics_file}")
            return np.array(metrics)
        except Exception as e:
            self.logger.error(f"Error loading metrics from {metrics_file}: {e}")
            return None
    
    def build_connectome_matrix(self, labels, weights=None, symmetric=True):
        """
        Build connectome matrix from streamline labels and optional weights
        
        Args:
            labels: Array of streamline labels (symmetric encoding)
            weights: Optional array of weights (e.g., FA values) for each streamline
            symmetric: Whether to create symmetric matrix
            
        Returns:
            connectome_matrix: Square matrix of size (num_labels, num_labels)
        """
        # Initialize connectome matrix
        connectome_matrix = np.zeros((self.num_labels, self.num_labels))
        
        # Convert symmetric labels to (i,j) pairs
        label_pairs = self._symmetric_to_pairs(labels)
        
        if weights is None:
            # Count-based connectome (Number of Streamlines - NoS)
            for i, j in label_pairs:
                if 0 <= i < self.num_labels and 0 <= j < self.num_labels:
                    connectome_matrix[i, j] += 1
                    if symmetric and i != j:
                        connectome_matrix[j, i] += 1
        else:
            # Weighted connectome (e.g., mean FA, MD, AD, RD)
            for idx, (i, j) in enumerate(label_pairs):
                if 0 <= i < self.num_labels and 0 <= j < self.num_labels and idx < len(weights):
                    # For weighted connectomes, we typically want the mean value per connection
                    # So we need to track both sum and count
                    if connectome_matrix[i, j] == 0:
                        connectome_matrix[i, j] = weights[idx]
                    else:
                        # Average with existing values
                        connectome_matrix[i, j] = (connectome_matrix[i, j] + weights[idx]) / 2
                    
                    if symmetric and i != j:
                        if connectome_matrix[j, i] == 0:
                            connectome_matrix[j, i] = weights[idx]
                        else:
                            connectome_matrix[j, i] = (connectome_matrix[j, i] + weights[idx]) / 2
        
        return connectome_matrix
    
    def _symmetric_to_pairs(self, symmetric_labels):
        """Convert symmetric labels to (i,j) pairs"""
        pairs = []
        for label in symmetric_labels:
            # Convert symmetric encoding back to (i,j) pairs
            # This assumes symmetric encoding where label = i*num_labels + j for i <= j
            if label >= 0 and label < self.num_labels * self.num_labels:
                i = label // self.num_labels
                j = label % self.num_labels
                pairs.append((i, j))
        return pairs
    
    def save_connectome(self, connectome_matrix, filename):
        """Save connectome matrix to CSV file"""
        try:
            output_path = os.path.join(self.out_path, filename)
            np.savetxt(output_path, connectome_matrix, delimiter=',', fmt='%.6f')
            self.logger.info(f"Saved connectome matrix to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving connectome to {filename}: {e}")
            return None


class ConnectomeComparator:
    """
    Class to compare predicted and ground truth connectomes
    """
    
    def __init__(self, out_path, logger=None):
        self.out_path = out_path
        self.logger = logger if logger else self._create_default_logger()
        self.results = {}
    
    def _create_default_logger(self):
        """Create a default logger if none provided"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def load_connectome(self, filepath):
        """Load connectome matrix from CSV file"""
        try:
            connectome = np.loadtxt(filepath, delimiter=',')
            self.logger.info(f"Loaded connectome from {filepath} with shape {connectome.shape}")
            return connectome
        except Exception as e:
            self.logger.error(f"Error loading connectome from {filepath}: {e}")
            return None
    
    def compute_correlation(self, pred_connectome, true_connectome, method='pearson'):
        """
        Compute correlation between predicted and true connectomes
        
        Args:
            pred_connectome: Predicted connectome matrix
            true_connectome: Ground truth connectome matrix
            method: 'pearson' or 'spearman'
            
        Returns:
            correlation coefficient and p-value
        """
        # Flatten matrices and remove diagonal (self-connections)
        mask = ~np.eye(pred_connectome.shape[0], dtype=bool)
        pred_flat = pred_connectome[mask].flatten()
        true_flat = true_connectome[mask].flatten()
        
        # Remove zero connections if needed
        nonzero_mask = (pred_flat != 0) | (true_flat != 0)
        pred_flat = pred_flat[nonzero_mask]
        true_flat = true_flat[nonzero_mask]
        
        if method == 'pearson':
            corr, p_value = pearsonr(pred_flat, true_flat)
        elif method == 'spearman':
            corr, p_value = spearmanr(pred_flat, true_flat)
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
        return corr, p_value
    
    def compute_lerm(self, pred_connectome, true_connectome):
        """
        Compute Linear Error in Relative Magnitude (LERM)
        LERM = |pred - true| / (pred + true) * 2
        """
        # Avoid division by zero
        denominator = pred_connectome + true_connectome
        mask = denominator != 0
        
        lerm = np.zeros_like(pred_connectome)
        lerm[mask] = np.abs(pred_connectome[mask] - true_connectome[mask]) / denominator[mask] * 2
        
        # Return mean LERM (excluding diagonal)
        mask_no_diag = ~np.eye(pred_connectome.shape[0], dtype=bool)
        mean_lerm = np.mean(lerm[mask_no_diag])
        
        return mean_lerm, lerm
    
    def compute_error_metrics(self, pred_connectome, true_connectome):
        """Compute various error metrics"""
        mask = ~np.eye(pred_connectome.shape[0], dtype=bool)
        pred_flat = pred_connectome[mask].flatten()
        true_flat = true_connectome[mask].flatten()
        
        mae = mean_absolute_error(true_flat, pred_flat)
        mse = mean_squared_error(true_flat, pred_flat)
        rmse = np.sqrt(mse)
        
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    
    def compare_connectomes(self, pred_connectome, true_connectome, comparison_name):
        """
        Comprehensive comparison between predicted and true connectomes
        """
        results = {}
        
        # Correlation analysis
        pearson_r, pearson_p = self.compute_correlation(pred_connectome, true_connectome, 'pearson')
        spearman_r, spearman_p = self.compute_correlation(pred_connectome, true_connectome, 'spearman')
        
        results['pearson_r'] = pearson_r
        results['pearson_p'] = pearson_p
        results['spearman_r'] = spearman_r
        results['spearman_p'] = spearman_p
        
        # LERM analysis
        mean_lerm, lerm_matrix = self.compute_lerm(pred_connectome, true_connectome)
        results['mean_lerm'] = mean_lerm
        
        # Error metrics
        error_metrics = self.compute_error_metrics(pred_connectome, true_connectome)
        results.update(error_metrics)
        
        # Store results
        self.results[comparison_name] = results
        
        # Log results
        self.logger.info(f"Comparison results for {comparison_name}:")
        self.logger.info(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
        self.logger.info(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4e})")
        self.logger.info(f"  Mean LERM: {mean_lerm:.4f}")
        self.logger.info(f"  MAE: {error_metrics['MAE']:.4f}")
        self.logger.info(f"  RMSE: {error_metrics['RMSE']:.4f}")
        
        return results
    
    def plot_comparison(self, pred_connectome, true_connectome, comparison_name, save_plot=True):
        """
        Create comparison plots between predicted and true connectomes
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Connectome Comparison: {comparison_name}', fontsize=16)
        
        # Plot 1: True connectome
        im1 = axes[0, 0].imshow(true_connectome, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Ground Truth Connectome')
        axes[0, 0].set_xlabel('Region')
        axes[0, 0].set_ylabel('Region')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Predicted connectome
        im2 = axes[0, 1].imshow(pred_connectome, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Predicted Connectome')
        axes[0, 1].set_xlabel('Region')
        axes[0, 1].set_ylabel('Region')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: Scatter plot
        mask = ~np.eye(pred_connectome.shape[0], dtype=bool)
        pred_flat = pred_connectome[mask].flatten()
        true_flat = true_connectome[mask].flatten()
        
        axes[1, 0].scatter(true_flat, pred_flat, alpha=0.5, s=1)
        axes[1, 0].plot([0, max(true_flat.max(), pred_flat.max())], 
                        [0, max(true_flat.max(), pred_flat.max())], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('Ground Truth')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title('Scatter Plot')
        
        # Add correlation info
        if comparison_name in self.results:
            r = self.results[comparison_name]['pearson_r']
            axes[1, 0].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[1, 0].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 4: Difference matrix
        diff_matrix = pred_connectome - true_connectome
        im4 = axes[1, 1].imshow(diff_matrix, cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('Difference (Pred - True)')
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('Region')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.out_path, f'connectome_comparison_{comparison_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {plot_path}")
        
        return fig
    
    def save_results_summary(self, filename='connectome_comparison_summary.csv'):
        """Save all comparison results to a CSV file"""
        if not self.results:
            self.logger.warning("No comparison results to save")
            return
        
        df = pd.DataFrame.from_dict(self.results, orient='index')
        output_path = os.path.join(self.out_path, filename)
        df.to_csv(output_path)
        self.logger.info(f"Saved comparison summary to {output_path}")
        return output_path


def create_all_connectomes(subject_path, atlas, pred_labels_file, true_labels_file, 
                          diffusion_metrics_dir, out_path, num_labels, logger=None):
    """
    Create all types of connectomes (NoS, FA, MD, AD, RD) for both predicted and true labels
    
    Args:
        subject_path: Path to subject directory
        atlas: Atlas name (e.g., 'aparc+aseg')
        pred_labels_file: Path to predicted labels file
        true_labels_file: Path to true labels file
        diffusion_metrics_dir: Directory containing diffusion metric files
        out_path: Output directory
        num_labels: Number of labels for the atlas
        logger: Logger instance
    """
    if logger is None:
        logger = create_logger(out_path)
    
    # Initialize connectome builder
    builder = ConnectomeBuilder(num_labels, out_path, logger)
    
    # Load labels
    pred_labels = builder.load_streamline_labels(pred_labels_file)
    true_labels = builder.load_streamline_labels(true_labels_file)
    
    if pred_labels is None or true_labels is None:
        logger.error("Failed to load labels")
        return None
    
    # Diffusion metric files
    metric_files = {
        'fa': os.path.join(diffusion_metrics_dir, 'mean_fa_per_streamline.txt'),
        'md': os.path.join(diffusion_metrics_dir, 'mean_md_per_streamline.txt'),
        'ad': os.path.join(diffusion_metrics_dir, 'mean_ad_per_streamline.txt'),
        'rd': os.path.join(diffusion_metrics_dir, 'mean_rd_per_streamline.txt')
    }
    
    connectome_files = {}
    
    # Create NoS (Number of Streamlines) connectomes
    logger.info("Creating NoS connectomes...")
    pred_nos = builder.build_connectome_matrix(pred_labels)
    true_nos = builder.build_connectome_matrix(true_labels)
    
    connectome_files['pred_nos'] = builder.save_connectome(pred_nos, f'connectome_pred_nos_{atlas}.csv')
    connectome_files['true_nos'] = builder.save_connectome(true_nos, f'connectome_true_nos_{atlas}.csv')
    
    # Create weighted connectomes for each diffusion metric
    for metric_name, metric_file in metric_files.items():
        if os.path.exists(metric_file):
            logger.info(f"Creating {metric_name.upper()}-weighted connectomes...")
            
            # Load diffusion metric values
            metric_values = builder.load_diffusion_metrics(metric_file)
            
            if metric_values is not None:
                # Create weighted connectomes
                pred_weighted = builder.build_connectome_matrix(pred_labels, metric_values)
                true_weighted = builder.build_connectome_matrix(true_labels, metric_values)
                
                connectome_files[f'pred_{metric_name}'] = builder.save_connectome(
                    pred_weighted, f'connectome_pred_{metric_name}_{atlas}.csv')
                connectome_files[f'true_{metric_name}'] = builder.save_connectome(
                    true_weighted, f'connectome_true_{metric_name}_{atlas}.csv')
        else:
            logger.warning(f"Diffusion metric file not found: {metric_file}")
    
    return connectome_files


def compare_all_connectomes(connectome_files, out_path, atlas, logger=None):
    """
    Compare all predicted vs true connectomes and generate analysis
    """
    if logger is None:
        logger = create_logger(out_path)
    
    comparator = ConnectomeComparator(out_path, logger)
    
    # Metrics to compare
    metrics = ['nos', 'fa', 'md', 'ad', 'rd']
    
    for metric in metrics:
        pred_key = f'pred_{metric}'
        true_key = f'true_{metric}'
        
        if pred_key in connectome_files and true_key in connectome_files:
            # Load connectomes
            pred_connectome = comparator.load_connectome(connectome_files[pred_key])
            true_connectome = comparator.load_connectome(connectome_files[true_key])
            
            if pred_connectome is not None and true_connectome is not None:
                # Compare connectomes
                comparison_name = f'{metric}_{atlas}'
                results = comparator.compare_connectomes(pred_connectome, true_connectome, comparison_name)
                
                # Create comparison plots
                comparator.plot_comparison(pred_connectome, true_connectome, comparison_name)
    
    # Save summary of all comparisons
    comparator.save_results_summary(f'connectome_comparison_summary_{atlas}.csv')
    
    return comparator.results