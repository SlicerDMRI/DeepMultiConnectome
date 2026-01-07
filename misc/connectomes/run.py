import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.linear_model import LinearRegression
from scipy.linalg import logm, norm
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import permutation_test
from sklearn.utils import resample
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

class ConnectomeData:
    def __init__(self, folder_path_test, output_folder, folder_path_retest=None):
        self.folder_path_test = folder_path_test
        self.output_folder = output_folder
        self.folder_path_retest = folder_path_retest
        self.data = {}
        self._load_data()

    def _load_data(self):
        for file_name in os.listdir(self.folder_path_test):
            if file_name.endswith(".csv"):
                parts = file_name.split("_")
                if len(parts) < 4:
                    continue

                subject_id = parts[0]
                atlas = parts[2]
                connectome_type = parts[3].replace(".csv", "")

                if subject_id not in self.data:
                    self.data[subject_id] = {}
                    if self.folder_path_retest is not None:
                        self.data[subject_id+"_retest"] = {}
                if atlas not in self.data[subject_id]:
                    self.data[subject_id][atlas] = {}
                    if self.folder_path_retest is not None:
                        self.data[subject_id+"_retest"][atlas] = {}

                file_path = os.path.join(self.folder_path_test, file_name)
                
                # Read test data                
                self.data[subject_id][atlas][connectome_type] = pd.read_csv(
                    file_path, header=None).values
                
                # If retest folder is provided, load retest data as well
                if self.folder_path_retest is not None:
                    file_path_retest = os.path.join(
                        self.folder_path_retest, file_name)
                    if os.path.exists(file_path_retest):
                        self.data[subject_id+"_retest"][atlas][connectome_type] = pd.read_csv(
                            file_path_retest, header=None).values

    def get_connectome(self, subject_id, atlas, connectome_type, upper=False):
        try:
            matrix = self.data[subject_id][atlas][connectome_type]
            if upper:
                # k=0 includes diagonal
                return matrix[np.triu_indices_from(matrix, k=0)]
            else:
                return matrix

        except KeyError:
            raise KeyError(
                f"Connectome for subject '{subject_id}', atlas '{atlas}', and type '{connectome_type}' not found.")

    def difference_map(self, atlas):
            # Compute the difference map between the predicted and true connectomes
        difference_maps = {}
        for subject_id in self.data:
            if "_retest" not in subject_id:
                if atlas in self.data[subject_id]:
                    true = self.get_connectome(subject_id, atlas, "true")
                    pred = self.get_connectome(subject_id, atlas, "pred")
                    difference_maps[subject_id] = true - pred

        # Compute the average difference map
        avg_difference_map = np.mean(list(difference_maps.values()), axis=0)

        # Plot the average difference map
        output_file = os.path.join(
            self.output_folder, f'{atlas}_average_difference_map.png')
        self.plot_connectome(avg_difference_map, output_file,
                             f'Average Difference Map for Atlas {atlas}', log_scale=False, difference=True)
        print(f"Average difference map saved as {output_file}")

        return difference_maps

    def plot_connectome(self, connectome_matrix, output_file, title, log_scale, difference=False):
        if difference == True:
            if not log_scale:
                cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
                norm = mcolors.TwoSlopeNorm(
                    vmin=connectome_matrix.min(), vcenter=0, vmax=connectome_matrix.max())
            if log_scale:
                cmap = plt.get_cmap('Reds')  # Red-white-blue colormap
                norm = mcolors.LogNorm(
                    vmin=max(connectome_matrix.min(), 1), vmax=connectome_matrix.max())
                connectome_matrix = np.where(
                    connectome_matrix == 0, 1e-6, connectome_matrix)
        elif difference == 'percent':
            cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
            norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        elif difference == 'accuracy':
            cmap = plt.get_cmap('BuGn')
            norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        else:
            cmap = plt.get_cmap('jet')

            if log_scale:
                norm = mcolors.LogNorm(
                    vmin=max(connectome_matrix.min(), 1), vmax=connectome_matrix.max())
                # Handle zero values by replacing them with a small positive value for log scale
                connectome_matrix = np.where(
                    connectome_matrix == 0, 1e-6, connectome_matrix)
            else:
                norm = None

        # Plot the connectome matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(connectome_matrix, cmap=cmap, norm=norm)
        plt.colorbar(label='Connection Strength')
        plt.title(title)
        plt.xlabel('node')
        plt.ylabel('node')
        num_nodes = connectome_matrix.shape[0]
        plt.xticks(ticks=np.arange(9, num_nodes, 10),
                   labels=np.arange(10, num_nodes, 10))
        plt.yticks(ticks=np.arange(9, num_nodes, 10),
                   labels=np.arange(10, num_nodes, 10))

        # Save the plot as an image file
        plt.savefig(output_file, bbox_inches='tight', dpi=500)
        plt.close()
    
    def compute_metrics(self, atlas, is_trt=False):
        metrics_true = {}
        metrics_pred = {}

        # Determine file paths based on whether TRT is used
        metrics_file = os.path.join(self.output_folder, f"{atlas}_metrics.csv")
        metrics_true_file = os.path.join(
            self.output_folder, f"{atlas}_metrics_TRT_true.csv")
        metrics_pred_file = os.path.join(
            self.output_folder, f"{atlas}_metrics_TRT_pred.csv")

        # Check if metrics have already been computed and saved
        if not is_trt:
            if os.path.exists(metrics_file):
                print(f"Metrics already computed. Loading from {metrics_file}")
                return pd.read_csv(metrics_file).to_dict()  # Convert to dict
        else:
            if os.path.exists(metrics_true_file) and os.path.exists(metrics_pred_file):
                print(
                    f"Metrics already computed. Loading from {metrics_true_file} and {metrics_pred_file}")
                # Convert to dict
                return pd.read_csv(metrics_true_file).to_dict(), pd.read_csv(metrics_pred_file).to_dict()

        # Compute intra and inter-subject metrics
        for subject_id in self.data:
            if "_retest" not in subject_id:
                metrics_true[subject_id] = {
                    "intra_LERM": np.nan,
                    "intra_r": np.nan,
                    "inter_LERM": np.nan,
                    "inter_r": np.nan,
                    "inter_LERM_all": [],
                    "inter_r_all": []
                }
                metrics_pred[subject_id] = {
                    "intra_LERM": np.nan,
                    "intra_r": np.nan,
                    "inter_LERM": np.nan,
                    "inter_r": np.nan,
                    "inter_LERM_all": [],
                    "inter_r_all": []
                }

                if atlas in self.data[subject_id]:
                        if is_trt:  # TRT case
                            true_T = self.get_connectome(subject_id, atlas, "true")
                            pred_T = self.get_connectome(subject_id, atlas, "pred")
                            true_RT = self.get_connectome(subject_id+"_retest", atlas, "true")
                            pred_RT = self.get_connectome(subject_id+"_retest", atlas, "pred")

                            # Intra-subject: distance between predicted and true for the same subject
                            metrics_true[subject_id]["intra_LERM"] = norm(logm(true_T) - logm(true_RT), 'fro')
                            metrics_pred[subject_id]["intra_LERM"] = norm(logm(pred_T) - logm(pred_RT), 'fro')
                            metrics_true[subject_id]["intra_r"] = np.corrcoef(true_T[np.triu_indices_from(true_T, k=0)], true_RT[np.triu_indices_from(true_RT, k=0)])[1, 0]
                            metrics_pred[subject_id]["intra_r"] = np.corrcoef(pred_T[np.triu_indices_from(pred_T, k=0)], pred_RT[np.triu_indices_from(pred_RT, k=0)])[1, 0]

                            # Inter-subject: distance between predicted connectome of one subject and true connectome of all other subjects
                            for other_id in self.data:
                                if other_id != subject_id and atlas in self.data[other_id] and "_retest" not in other_id:
                                    true_RT_other = self.get_connectome(other_id+"_retest", atlas, "true")
                                    pred_RT_other = self.get_connectome(other_id+"_retest", atlas, "true")
                                    # Append inter-subject values instead of overwriting
                                    metrics_true[subject_id]["inter_LERM_all"].append(norm(logm(true_T) - logm(true_RT_other), 'fro'))
                                    metrics_pred[subject_id]["inter_LERM_all"].append(norm(logm(pred_T) - logm(pred_RT_other), 'fro'))
                                    metrics_true[subject_id]["inter_r_all"].append(np.corrcoef(true_T[np.triu_indices_from(true_T, k=0)], true_RT_other[np.triu_indices_from(true_RT_other, k=0)])[1, 0])
                                    metrics_pred[subject_id]["inter_r_all"].append(np.corrcoef(pred_T[np.triu_indices_from(pred_T, k=0)], pred_RT_other[np.triu_indices_from(pred_RT_other, k=0)])[1, 0])

                            # Compute mean inter values for printing
                            metrics_true[subject_id]["inter_LERM"] = np.mean(metrics_true[subject_id]["inter_LERM_all"])
                            metrics_true[subject_id]["inter_r"] = np.mean(metrics_true[subject_id]["inter_r_all"])
                            metrics_pred[subject_id]["inter_LERM"] = np.mean(metrics_pred[subject_id]["inter_LERM_all"])
                            metrics_pred[subject_id]["inter_r"] = np.mean(metrics_pred[subject_id]["inter_r_all"])

                        else:  # Non-TRT case
                            true = self.get_connectome(subject_id, atlas, "true")
                            pred = self.get_connectome(subject_id, atlas, "pred")

                            # Intra-subject: distance between predicted and true for the same subject
                            metrics_true[subject_id]["intra_LERM"] = norm(logm(pred) - logm(true), 'fro')
                            metrics_true[subject_id]["intra_r"] = np.corrcoef(pred[np.triu_indices_from(pred, k=0)], true[np.triu_indices_from(true, k=0)])[1, 0]
                            metrics_true[subject_id]["intra_MAE"] = mean_absolute_error(true.flatten(), pred.flatten())
                            metrics_true[subject_id]["intra_MSE"] = mean_squared_error(true.flatten(), pred.flatten())
                            metrics_true[subject_id]["intra_RMSE"] = np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))
                            metrics_true[subject_id]["intra_R2"] = r2_score(true.flatten(), pred.flatten())
                            metrics_true[subject_id]["intra_MAPE"] = mean_absolute_percentage_error(true.flatten(), pred.flatten())
                            
                            # Inter-subject: distance between predicted connectome of one subject and true connectome of all other subjects
                            for other_id in self.data:
                                if other_id != subject_id and atlas in self.data[other_id]:
                                    true_other = self.get_connectome(other_id, atlas, "true")
                                    # Append inter-subject values instead of overwriting #! inter is ground truth connectomes between different subjects
                                    metrics_true[subject_id]["inter_LERM_all"].append(norm(logm(true) - logm(true_other), 'fro'))
                                    metrics_true[subject_id]["inter_r_all"].append(np.corrcoef(true[np.triu_indices_from(true, k=0)], true_other[np.triu_indices_from(true_other, k=0)])[1, 0])
                                    metrics_true[subject_id]["inter_MAE_all"] = mean_absolute_error(true.flatten(), true_other.flatten())
                                    metrics_true[subject_id]["inter_MSE_all"] = mean_squared_error(true.flatten(), true_other.flatten())
                                    metrics_true[subject_id]["inter_RMSE_all"] = np.sqrt(mean_squared_error(true.flatten(), true_other.flatten()))
                                    metrics_true[subject_id]["inter_R2_all"] = r2_score(true.flatten(), true_other.flatten())
                                    metrics_true[subject_id]["inter_MAPE_all"] = mean_absolute_percentage_error(true.flatten(), true_other.flatten())
                                    
                            # Compute mean inter values for printing
                            metrics_true[subject_id]["inter_LERM"] = np.mean(metrics_true[subject_id]["inter_LERM_all"])
                            metrics_true[subject_id]["inter_r"] = np.mean(metrics_true[subject_id]["inter_r_all"])
                            metrics_true[subject_id]["inter_MAE"] = np.mean(metrics_true[subject_id]["inter_MAE_all"])
                            metrics_true[subject_id]["inter_MSE"] = np.mean(metrics_true[subject_id]["inter_MSE_all"])
                            metrics_true[subject_id]["inter_RMSE"] = np.mean(metrics_true[subject_id]["inter_RMSE_all"])
                            metrics_true[subject_id]["inter_R2"] = np.mean(metrics_true[subject_id]["inter_R2_all"])
                            metrics_true[subject_id]["inter_MAPE"] = np.mean(metrics_true[subject_id]["inter_MAPE_all"])
                            
                            print(metrics_true[subject_id]["inter_MAPE"], metrics_true[subject_id]["inter_MAPE"])
                        print(f"Subject {subject_id} done")

        if not is_trt:  # Non-TRT case
            metrics_df = pd.DataFrame.from_dict(metrics_true, orient='index')
            metrics_df.to_csv(metrics_file)
            return metrics_true  # Returning the metrics dictionary
        else:  # TRT case
            metrics_true_df = pd.DataFrame.from_dict(metrics_true, orient="index")
            metrics_true_df.to_csv(metrics_true_file)
            metrics_pred_df = pd.DataFrame.from_dict(metrics_pred, orient="index")
            metrics_pred_df.to_csv(metrics_pred_file)
            return metrics_true, metrics_pred  # Returning both true and predicted metrics


    def analyze_metrics(self, atlas, mode=""):
        metrics_file = os.path.join(self.output_folder, f"{atlas}_metrics{mode}.csv")

        if not os.path.exists(metrics_file):
            print(f"Metrics file for {atlas} not found!")
            return

        # Load metrics from file
        metrics_df = pd.read_csv(metrics_file, index_col=0)

        p_values = []  # To store p-values for FDR correction
        metrics_for_fdr = []  # Store which metrics the p-values correspond to
        
        # Extract metric names from the columns        
        metric_names = []
        for metric in metrics_df.columns:
            metric_name = metric.split("_")[1]
            if metric_name not in metric_names:
                metric_names.append(metric_name)
        
        print(f"\n==========Statistical Analysis for {atlas} atlas==========")
        for metric in metric_names:
            intra = metrics_df['intra_'+metric]
            inter = metrics_df['inter_'+metric]
            
            # Descriptive statistics
            print("\nDescriptive Statistics:")
            print(f"{metric}: Intra: {np.mean(intra):.3f}±{np.std(intra, ddof=1):.3f}, Inter: {np.mean(inter):.3f}±{np.std(inter, ddof=1):.3f}")

            # Skewness and Kurtosis
            print(
                f"{metric}: Intra - Skewness: {stats.skew(intra):.3f}, Kurtosis: {stats.kurtosis(intra):.3f}")
            print(
                f"{metric}: Inter - Skewness: {stats.skew(inter):.3f}, Kurtosis: {stats.kurtosis(inter):.3f}")

            # Normality test (Shapiro-Wilk)
            print(
                f"{metric}: Intra - Shapiro-Wilk Test: W = {stats.shapiro(intra)[0]:.3f}, p = {stats.shapiro(intra)[1]:.5f}")
            print(
                f"{metric}: Inter - Shapiro-Wilk Test: W = {stats.shapiro(inter)[0]:.3f}, p = {stats.shapiro(inter)[1]:.5f}")

            # Statistical Analysis
            print(
                "\nStatistical Analysis (Paired t-test, Cohen's d, Percentage Improvement):")
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(intra, inter)
            df = len(intra) - 1
            print(f"{metric}: t({df}) = {t_stat:.3f}, p = {p_val:.5f}")

            # Collect p-values for FDR correction
            p_values.append(p_val)
            metrics_for_fdr.append(metric)

            # Effect size (Cohen's d)
            d = (np.mean(intra) - np.mean(inter)) / \
                np.std(intra - inter, ddof=1)
            print(f"Effect Size (Cohen's d): {metric}: d = {d:.3f}")

            # Percentage improvement
            improvement = (np.mean(intra) - np.mean(inter)) / \
                np.mean(inter) * 100
            print(f"Percentage Improvement ({metric}): {improvement:.2f}%")

            # Interpretation of significance (Before FDR correction)
            if p_val < 0.05:
                print(
                    f"Significant difference found between intra and inter {metric}.")
            else:
                print(
                    f"No significant difference between intra and inter {metric}.")

        if False:
            # Apply FDR correction to all p-values
            rejected, corrected_p_values, _, _ = multipletests(
                p_values, alpha=0.05, method='fdr_bh')

            # Adjust significance interpretation based on corrected p-values
            for i, metric in enumerate(metrics_for_fdr):
                corrected_p_val = corrected_p_values[i]
                print(f"\n{metric} (Corrected for FDR):")
                if corrected_p_val < 0.05:
                    print(
                        f"Significant difference found after FDR correction, corrected p = {corrected_p_val:.5f}")
                else:
                    print(
                        f"No significant difference after FDR correction, corrected p = {corrected_p_val:.5f}")
        
        
    def plot_double_violin(self, atlases, mode="",n_rows = 1):
        if isinstance(atlases, str):
            atlases = [atlases]

        atlas_names = {"aparc+aseg": "84 ROIs", "aparc.a2009s+aseg": "164 ROIs"}  # Rename for plotting

        plot_df_list = []
        metric_names = set()
        p_vals = {}
        p_vals = {atlas: {} for atlas in atlases}

        for atlas in atlases:
            metrics_file = os.path.join(self.output_folder, f"{atlas}_metrics{mode}.csv")

            if not os.path.exists(metrics_file):
                print(f"Metrics file for {atlas} not found!")
                continue

            metrics_df = pd.read_csv(metrics_file, index_col=0)
            for col in metrics_df.columns:
                if col.startswith("intra_"):
                    metric_name = col.split("_")[1]
                    metric_names.add(metric_name)

            for metric in metric_names:
                intra = metrics_df[f'intra_{metric}']
                inter = metrics_df[f'inter_{metric}']
                
                # Normality check with Shapiro-Wilk test
                normal = stats.shapiro(intra)[1] > 0.05 and stats.shapiro(inter)[1] > 0.05

                # Check normality and apply appropriate statistical test
                if normal:
                    _, p_val = stats.ttest_rel(intra, inter)
                    annot_color = "b"
                else:
                    _, p_val = stats.wilcoxon(intra, inter)
                    annot_color = "k"          
                
                p_vals[atlas][metric] = p_val

                plot_df = pd.DataFrame({
                    'Type': ['Intrasubject'] * len(intra) + ['Intersubject'] * len(inter),
                    'Value': list(intra) + list(inter),
                    'Atlas': [atlas_names[atlas]] * (len(intra) + len(inter)),
                    'Metric': [metric] * (len(intra) + len(inter))
                })
                plot_df_list.append(plot_df)


        combined_df = pd.concat(plot_df_list, ignore_index=True)

        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'axes.labelsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16
        })

        unique_metrics = sorted(combined_df['Metric'].unique())
        unique_metrics = ['r', 'LERM']
        fig, axes = plt.subplots(n_rows, len(unique_metrics) // n_rows, figsize=(7 * (len(unique_metrics) // n_rows), 6*n_rows))
        axes = axes.flatten()
        
        if len(unique_metrics) == 1:
            axes = [axes]

        palette = ["#699cc7", "#ed524e"]

        for i, (ax, metric) in enumerate(zip(axes, unique_metrics)):
            metric_df = combined_df[combined_df['Metric'] == metric]
            sns.violinplot(x='Atlas', y='Value', data=metric_df, hue='Type', split=True, 
                        inner="quart", ax=ax, legend=True, palette=palette)
            title = metric.replace("_", " ") if metric != "r" else "\033[3mr"
            ax.set_xlabel('Parcellation scheme', fontsize=18)
            ax.set_ylabel(metric.replace("_", " "), fontsize=18)
            title = metric.replace("_", " ") if metric != "r" else "Pearson's correlation"
            ax.set_title(title, fontsize=22)
            ax.legend(loc='lower right')
            
            for j, atlas in enumerate(atlases):
                if metric in ['r', 'R2']:
                    y_max =  0.9995
                else:
                    y_max = metric_df[metric_df['Atlas'] == atlas_names[atlas]]['Value'].max() + 0.75
                p_val = p_vals[atlas][metric]
                significance = "n.s." if p_val >= 0.05 else ("*" if p_val >= 0.01 else ("**" if p_val >= 0.001 else "***"))
                ax.annotate(significance, xy=(j, y_max), ha='center', fontsize=16, color='black')

        plt.tight_layout()
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().set_title("")
        plt.show()
        
    def load_metrics(self, output_folder, atlas):
        """Load and organize metric data from CSV files."""
        metrics_file_true = os.path.join(output_folder, f"{atlas}_metrics_TRT_true.csv")
        metrics_file_pred = os.path.join(output_folder, f"{atlas}_metrics_TRT_pred.csv")
        
        metrics_df_true = pd.read_csv(metrics_file_true, index_col=0)
        metrics_df_pred = pd.read_csv(metrics_file_pred, index_col=0)
        
        data = {}
        for metric in ['intra_LERM', 'intra_r', 'inter_LERM', 'inter_r']:
            data[f"true_{metric}"] = metrics_df_true[metric]
            data[f"pred_{metric}"] = metrics_df_pred[metric]
        
        del metrics_df_true, metrics_df_pred
        gc.collect()
        
        return data

    def prepare_plot_data(self, data, atlas_label):
        """Prepare a single dataframe for violin plotting."""
        df_list = []
        for metric in ['LERM', 'r']:
            num_samples = len(data[f'true_intra_{metric}'])
            
            df = pd.DataFrame({
                'Type': (['Ground truth intrasubject'] * num_samples + 
                        ['Predicted intrasubject'] * num_samples + 
                         ['Ground truth intersubject'] * num_samples +
                        ['Predicted intersubject'] * num_samples),
                'Value': np.concatenate([
                    data[f'true_intra_{metric}'], data[f'pred_intra_{metric}'],
                    data[f'true_inter_{metric}'], data[f'pred_inter_{metric}']
                ]),
                'Atlas': [atlas_label] * (4 * num_samples)
            })
            df_list.append(df)
        return df_list


    def plot_double_violin_TRT(self, atlases):
        if isinstance(atlases, str):
            atlases = [atlases]

        atlas_labels = {"aparc+aseg": "84 ROIs", "aparc.a2009s+aseg": "164 ROIs"}
        plot_data_LERM, plot_data_r = [], []

        for atlas in atlases:
            data = self.load_metrics(self.output_folder, atlas)
            df_LERM, df_r = self.prepare_plot_data(data, atlas_labels.get(atlas, atlas))
            plot_data_LERM.append(df_LERM)
            plot_data_r.append(df_r)
        
        plot_df_LERM_combined = pd.concat(plot_data_LERM, ignore_index=True)
        plot_df_r_combined = pd.concat(plot_data_r, ignore_index=True)
        
        palette_LERM = ["#699cc7", "#ed524e", "#d3d3d3", "#6a4c93"]
        palette_LERM = ["#699cc7", "#ed524e", "#f4d35e", "#7cb490"]
        palette_r = ["#699cc7", "#ed524e", "#f4d35e", "#7cb490"]

        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'axes.labelsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.violinplot(x='Atlas', y='Value', data=plot_df_r_combined, hue='Type', split=True, inner="quart", ax=axes[0], palette=palette_r)
        axes[0].set_title("Pearson's correlation", fontsize=22)
        axes[0].set_xlabel('Atlas', fontsize=18)
        axes[0].set_ylabel('$\it{r}$', fontsize=18)
        axes[0].legend(loc='lower right')

        sns.violinplot(x='Atlas', y='Value', data=plot_df_LERM_combined, hue='Type', split=True, inner="quart", ax=axes[1], palette=palette_LERM)
        axes[1].set_title("LERM", fontsize=22)
        axes[1].set_xlabel('Atlas', fontsize=18)
        axes[1].set_ylabel('LERM Distance', fontsize=18)
        axes[1].legend(loc='lower right')

        # Compute p-values
        p_vals_LERM, p_vals_r = [], []
        # for df_LERM, df_r in zip(plot_data_LERM, plot_data_r):
        #     for metric, p_vals in zip(['LERM', 'r'], [p_vals_LERM, p_vals_r]):
        true_intra_LERM, pred_intra_LERM = df_LERM[df_LERM['Type'] =='Ground truth intrasubject']['Value'], df_LERM[df_LERM['Type'] == 'Predicted intrasubject']['Value']
        true_inter_LERM, pred_inter_LERM = df_LERM[df_LERM['Type'] =='Ground truth intersubject']['Value'], df_LERM[df_LERM['Type'] == 'Predicted intersubject']['Value']
        p_vals_LERM.extend([stats.wilcoxon(true_intra_LERM, pred_intra_LERM)[1], stats.wilcoxon(true_inter_LERM, pred_inter_LERM)[1]])
        
        true_intra_r, pred_intra_r = df_r[df_r['Type'] == 'Ground truth intrasubject']['Value'], df_r[df_r['Type'] == 'Predicted intrasubject']['Value']
        true_inter_r, pred_inter_r = df_r[df_r['Type'] == 'Ground truth intersubject']['Value'], df_r[df_r['Type'] == 'Predicted intersubject']['Value']
        p_vals_r.extend([stats.wilcoxon(true_intra_r, pred_intra_r)[1], stats.wilcoxon(true_inter_r, pred_inter_r)[1]])
        
        reject_LERM, corrected_p_vals_LERM, _, _ = multipletests(p_vals_LERM, alpha=0.001, method='fdr_bh')
        reject_r, corrected_p_vals_r, _, _ = multipletests(p_vals_r, alpha=0.001, method='fdr_bh')
        
        
        # Fix annotation logic
        for ax, p_vals_metric, df, y_offset_factor in zip(axes, [corrected_p_vals_r, corrected_p_vals_LERM], [df_r, df_LERM], [1, 1.]):
            y_offset = df['Value'].max()#ax.get_ylim()[1] * y_offset_factor  # Dynamic y-offset
            # Get x positions from x-axis ticks
            x_positions = [tick.get_position()[0] for tick in ax.get_xticklabels()]

            for i, atlas in enumerate(atlases):
                for j, p_val in enumerate(p_vals_metric):
                    annotation_text = "n.s." if p_val >= 0.05 else (
                        "*" if p_val >= 0.01 else ("**" if p_val >= 0.001 else "***"))
                    # Adjust x position for each annotation
                    if j==0:
                        x_pos = x_positions[i] - 0.2
                    else:
                        x_pos = x_positions[i] + 0.2
                    # ax.annotate(annotation_text, xy=(x_pos, y_offset),ha='center', fontsize=16, color='k')

        plt.tight_layout()
        axes[0].get_legend().set_title("")
        axes[1].get_legend().set_title("")
        plt.show()


# Usage example

# ------------------ USAGE EXAMPLES ------------------

# 1. For predicted vs. true evaluation (original mode):
folder_path_test =  "D:/Code/ExternshipLocal/results/connectomes/"
output_folder =     "D:/Code/ExternshipLocal/results/metrics/"
connectome_data = ConnectomeData(folder_path_test, output_folder)

connectome_data.compute_metrics("aparc+aseg")
connectome_data.compute_metrics("aparc.a2009s+aseg")
# connectome_data.analyze_metrics("aparc+aseg")
# connectome_data.analyze_metrics("aparc.a2009s+aseg")
connectome_data.plot_double_violin(["aparc+aseg", "aparc.a2009s+aseg"])

# connectome_data.difference_map("aparc+aseg")


# 2. For a test–retest experiment:
folder_path_test =      "D:/code/ExternshipLocal/results/connectomes_test"
folder_path_retest =    "D:/code/ExternshipLocal/results/connectomes_retest"
output_folder =         "D:/Code/ExternshipLocal/results/metrics/"
connectome_data = ConnectomeData(folder_path_test, output_folder, folder_path_retest)

# connectome_data.compute_metrics("aparc+aseg", is_trt=True)
# connectome_data.compute_metrics("aparc.a2009s+aseg", is_trt=True)
# connectome_data.analyze_metrics("aparc+aseg", mode="_TRT_true")
# connectome_data.analyze_metrics("aparc+aseg", mode="_TRT_pred")
# connectome_data.analyze_metrics("aparc.a2009s+aseg", mode="_TRT_true")
# connectome_data.analyze_metrics("aparc.a2009s+aseg", mode="_TRT_pred")
# connectome_data.plot_double_violin_TRT(["aparc+aseg"])
# connectome_data.plot_double_violin_TRT(["aparc+aseg", "aparc.a2009s+aseg"])
