import sys
sys.path.append('..')
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, classification_report, accuracy_score, confusion_matrix, mean_absolute_error, mean_absolute_percentage_error 
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from utils.metrics_plots import classify_report, process_curves, calculate_acc_prec_recall_f1, best_swap, save_best_weights
from sklearn.metrics import precision_score, f1_score, recall_score
from numpy.linalg import norm
import networkx as nx
import pprint
from tractography.label_encoder import convert_labels_list

def create_connectome(labels, num_labels):
    connectome_matrix = np.zeros((num_labels, num_labels))
    if type(labels)==dict: # dict with tuples (nodes) as keys and scores as values
        for key, value in labels.items():
            x = key[0] 
            y = key[1]
            connectome_matrix[x, y]=value
            if x!=y:
                connectome_matrix[y, x]=value
    else:
        for i in range(len(labels) - 1):
            x=labels[i][0]
            y=labels[i][1]
            connectome_matrix[x, y] += 1
            if x!=y:
                connectome_matrix[y, x] += 1
    return connectome_matrix[1:,1:]

def save_connectome(connectome_matrix, out_path, title='true'):
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, f"connectome_{title}.csv")
    np.savetxt(csv_path, connectome_matrix, delimiter=',')

def plot_connectome(connectome_matrix, output_file, title, log_scale, difference=False):
    if difference==True:
        if not log_scale:
            cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
            norm = mcolors.TwoSlopeNorm(vmin=connectome_matrix.min(), vcenter=0, vmax=connectome_matrix.max())
        if log_scale:
            cmap = plt.get_cmap('Reds')  # Red-white-blue colormap
            norm = mcolors.LogNorm(vmin=max(connectome_matrix.min(), 1), vmax=connectome_matrix.max())
            connectome_matrix = np.where(connectome_matrix == 0, 1e-6, connectome_matrix)
    elif difference=='percent':
        cmap = plt.get_cmap('RdBu_r')  # Red-white-blue colormap
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    elif difference=='accuracy':
        cmap = plt.get_cmap('BuGn')
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    else:
        cmap = plt.get_cmap('jet')

        if log_scale:
            norm = mcolors.LogNorm(vmin=max(connectome_matrix.min(), 1), vmax=connectome_matrix.max())
            # Handle zero values by replacing them with a small positive value for log scale
            connectome_matrix = np.where(connectome_matrix == 0, 1e-6, connectome_matrix)
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
    plt.xticks(ticks=np.arange(9, num_nodes, 10), labels=np.arange(10, num_nodes, 10))
    plt.yticks(ticks=np.arange(9, num_nodes, 10), labels=np.arange(10, num_nodes, 10))
    
    # Save the plot as an image file
    plt.savefig(output_file, bbox_inches='tight', dpi=500)
    plt.close()

def label_wise_accuracy(true_labels, pred_labels):
    # Get the set of unique labels
    unique_labels = set(true_labels) | set(pred_labels)
    
    # Initialize dictionaries to store correct predictions and total counts
    correct_predictions = {label: 0 for label in unique_labels}
    total_counts = {label: 0 for label in unique_labels}
    
    # Iterate over true and predicted labels
    for true, pred in zip(true_labels, pred_labels):
        total_counts[true] += 1
        if true == pred:
            correct_predictions[true] += 1

    # Compute accuracy for each label
    label_accuracy = {}
    for label in unique_labels:
        if total_counts[label] > 0:
            label_accuracy[label] = correct_predictions[label] / total_counts[label]
        else:
            label_accuracy[label] = 1.0
    
    return label_accuracy

def label_wise_metrics(true_labels, pred_labels):
    unique_labels = set(true_labels) | set(pred_labels)
    
    accuracy = label_wise_accuracy(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, labels=[label], average='none', zero_division=0)
    recall = recall_score(true_labels, pred_labels, labels=[label], average='none', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, labels=[label], average='none', zero_division=0)
    
    return accuracy, precision, recall, f1

class ConnectomeMetrics:
    def __init__(self, true_labels=None, pred_labels=None, encoding='symmetric', atlas="aparc+aseg", out_path='output', graph=False, plot=True): # , state_labels_encoded=True
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.atlas = atlas
        self.encoding = encoding
        self.out_path = out_path
        self.results = {}
        
        num_labels_dict={"aparc+aseg":85, "aparc.a2009s+aseg":165}
        self.num_labels = num_labels_dict[atlas]
        
        # Decode labels from 1D list to 2D node pairs if necessary
        # if state_labels_encoded==True and self.true_labels!=None and self.pred_labels!=None:
        self.true_labels_decoded = convert_labels_list(self.true_labels, encoding_type=self.encoding, 
                                            mode='decode', num_labels=self.num_labels)
        self.pred_labels_decoded = convert_labels_list(self.pred_labels, encoding_type=self.encoding, 
                                            mode='decode', num_labels=self.num_labels)
        # elif state_labels_encoded==True and self.true_labels==None and self.pred_labels==None:
        #     print('Input labels to decode them')    
        
        self.true_connectome = create_connectome(self.true_labels_decoded, self.num_labels)
        self.pred_connectome = create_connectome(self.pred_labels_decoded, self.num_labels)

        # Save connectomes
        save_connectome(self.true_connectome, self.out_path, title=f'{self.atlas}_true')
        save_connectome(self.pred_connectome, self.out_path, title=f'{self.atlas}_pred')
        
        if plot:
            # Save different connectome plots
            self.plot_connectomes(zero_diagonal=False, log_scale=True)
            
            # Compute, save and plot alternate "connectomes"
            self.difference_conenctome()
            self.percentile_change_connectome()
            self.accuracy_connectome()
        
        self.compute_metrics()
        if graph:
            self.compute_network_metrics(version='true')
            self.compute_network_metrics(version='pred')
        
        # Save metrics
        results = pd.DataFrame([self.results])
        results.to_csv(os.path.join(self.out_path, f"metrics_{atlas}.csv"), index=False)
        print(f"Metrics saved to 'metrics_{atlas}.csv'")

        
    def plot_connectomes(self, zero_diagonal=False, log_scale=True):
        # Plots with diagonal    
        plot_connectome(self.true_connectome, f"{self.out_path}/{self.atlas}_true_logscaled.png", 
                        f"True connectome", log_scale=True)
        plot_connectome(self.pred_connectome, f"{self.out_path}/{self.atlas}_pred_logscaled.png", 
                        f"Predicted connectome", log_scale=True)
        # Plots with diagonal and not logscaled
        if log_scale==False:
            plot_connectome(self.true_connectome, f"{self.out_path}/{self.atlas}_true.png", 
                            f"True connectome", log_scale=False)
            plot_connectome(self.pred_connectome, f"{self.out_path}/{self.atlas}_pred.png", 
                            f"Predicted connectome", log_scale=False)

        # Plots without diagonal (set to 0)
        if zero_diagonal==True:
            plot_connectome(np.fill_diagonal(self.true_connectome, 0), f"{self.out_path}/{self.atlas}_true_logscaled_zerodiagonal.png", 
                            f"True connectome without diagonal", log_scale=True)
            plot_connectome(np.fill_diagonal(self.pred_connectome, 0), f"{self.out_path}/{self.atlas}_pred_logscaled_zerodiagonal.png", 
                            f"Predicted connectome without diagonal", log_scale=True)
            if log_scale==False:
                plot_connectome(np.fill_diagonal(self.true_connectome, 0), f"{self.out_path}/{self.atlas}_true_zerodiagonal.png", 
                                f"True connectome without diagonal", log_scale=False)
                plot_connectome(np.fill_diagonal(self.pred_connectome, 0), f"{self.out_path}/{self.atlas}_pred_zerodiagonal.png", 
                                f"Predicted connectome without diagonal", log_scale=False)
    
    
    def difference_conenctome(self, zero_diagonal=False):
        self.difference_connectome = self.true_connectome - self.pred_connectome
        
        save_connectome(self.difference_connectome, self.out_path, title=f'{self.atlas}_diff')

        # Plot with and without diagonal
        plot_connectome(self.difference_connectome, f"{self.out_path}/{self.atlas}_diff.png", 
                        f"Difference connectome (True-Predicted)", difference=True, log_scale=False)
        if zero_diagonal==True:
            plot_connectome(np.fill_diagonal(self.difference_connectome), f"{self.out_path}/{self.atlas}_diff_zerodiagonal.png", 
                            f"Difference connectome (True-Predicted)", difference=True, log_scale=False)
            
        self.difference_connectome_abs = np.absolute(self.difference_connectome)
        plot_connectome(self.difference_connectome_abs, f"{self.out_path}/{self.atlas}_diff_abs.png", 
                        f"Connectome absolute differences |True-Predicted|", difference=True, log_scale=True)
        
    def percentile_change_connectome(self, zero_diagonal=False):
        np.seterr(divide='ignore', invalid='ignore')
        percentchange_connectome = (self.true_connectome - self.pred_connectome) / self.true_connectome
        self.percentchange_connectome = np.nan_to_num(percentchange_connectome, nan=0.0, posinf=0.0, neginf=0.0)

        save_connectome(self.percentchange_connectome, self.out_path, title=f'{self.atlas}_perc')

        # Plot with and without diagonal
        plot_connectome(self.percentchange_connectome, f"{self.out_path}/{self.atlas}_perc.png", 
                        f"Percent change connectome ((True-Predicted)/True)", difference='percent', log_scale=False)
        if zero_diagonal==True:
            plot_connectome(self.percentchange_connectome, f"{self.out_path}/{self.atlas}_perc_zerodiagonal.png", 
                            f"Percent change connectome((True-Predicted)/True)", difference='percent', log_scale=False)

    def accuracy_connectome(self):
        # Compute accuracy per label
        accuracy_per_label_decoded = label_wise_accuracy(self.true_labels_decoded, self.pred_labels_decoded)
        self.acc_connectome=create_connectome(accuracy_per_label_decoded, self.num_labels)
        save_connectome(self.acc_connectome, self.out_path, title=f'{self.atlas}_acc')
        plot_connectome(self.acc_connectome, f"{self.out_path}/{self.atlas}_acc.png", 
                            f"Accuracy connectome", difference='accuracy', log_scale=False)

    def compute_metrics(self):
        
        
        # Edge overlap and other comparison metrics
        acc = accuracy_score(self.true_labels, self.pred_labels)
        mac_precision, mac_recall, mac_f1, support = precision_recall_fscore_support(self.true_labels, self.pred_labels, beta=1.0, average='macro', zero_division=np.nan) # ignore empty labels
        weighted_precision, weighted_recall, weighted_f1, support = precision_recall_fscore_support(self.true_labels, self.pred_labels, beta=1.0, average='weighted', zero_division=np.nan) # ignore empty labels
        
        # Correlation and similarity metrics
        pearson_corr, _ = pearsonr(self.true_connectome.flatten(), self.pred_connectome.flatten())
        spearman_corr, _ = spearmanr(self.true_connectome.flatten(), self.pred_connectome.flatten())
        mse = mean_squared_error(self.true_connectome, self.pred_connectome)
        mae = np.sum(np.absolute(self.true_connectome- self.pred_connectome))
        mape = mae/np.sum(self.true_connectome)

        # Frobenius Norm
        frobenius_norm = norm(self.true_connectome - self.pred_connectome, 'fro')

        # Earth Mover's Distance (Wasserstein distance)
        emd = wasserstein_distance(self.true_connectome.flatten(), self.pred_connectome.flatten())

        # IGNORING 0 LABEL
        ignore_labels = list(range(self.num_labels))
        mask = ~np.isin(self.true_labels, ignore_labels)
        filtered_labels = np.array(self.true_labels)[mask]
        filtered_predictions = np.array(self.pred_labels)[mask]
        Facc = accuracy_score(filtered_labels, filtered_predictions)
        Fmac_precision, Fmac_recall, Fmac_f1, Fsupport = precision_recall_fscore_support(filtered_labels, filtered_predictions, beta=1.0, average='macro', zero_division=np.nan) # ignore empty labels
        Fweighted_precision, Fweighted_recall, Fweighted_f1, Fsupport = precision_recall_fscore_support(filtered_labels, filtered_predictions, beta=1.0, average='weighted', zero_division=np.nan) # ignore empty labels
        
        # Storing the results
        metrics = {
            'Accuracy' : acc,
            'F1-Score (macro)': mac_f1,
            'Precision (macro)': mac_precision,
            'Recall (macro)': mac_recall,
            'F1-Score (weighted)': weighted_f1,
            'Precision (weighted)': weighted_precision,
            'Recall (weighted)': weighted_recall,
            'Accuracy without unknown' : Facc,
            'F1-Score (macro) without unknown': Fmac_f1,
            'Precision (macro) without unknown': Fmac_precision,
            'Recall (macro) without unknown': Fmac_recall,
            'F1-Score (weighted) without unknown': Fweighted_f1,
            'Precision (weighted) without unknown': Fweighted_precision,
            'Recall (weighted) without unknown': Fweighted_recall,
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'Pearson Correlation': pearson_corr,
            'Spearman Correlation': spearman_corr,
            'Frobenius Norm': frobenius_norm,
            'Earth Mover\'s Distance': emd
        }
        print(mape, mae)
        self.results.update(metrics)

    def global_reaching_centrality(self, G):
        """Compute Global Reaching Centrality (GRC) for a graph G."""
        reachability = {node: sum(nx.single_source_shortest_path_length(G, node).values()) / (len(G) - 1) for node in G}
        CmaxR = max(reachability.values())
        GRC = sum(CmaxR - reach for reach in reachability.values()) / (len(G) - 1)
        return GRC

    def small_worldness(self, G, niter=5):
        """Compute the small-worldness of a graph G using omega and sigma."""
        L = nx.average_shortest_path_length(G, weight='weight')
        C = nx.average_clustering(G, weight='weight')
        rand_graph = nx.random_reference(G, niter=niter)
        L_rand = nx.average_shortest_path_length(rand_graph, weight='weight')
        C_rand = nx.average_clustering(rand_graph, weight='weight')
        omega = L_rand / L - C / C_rand
        sigma = (C / C_rand) / (L / L_rand)
        return omega, sigma        
    # Transform weights: reciprocal of non-zero weights
    def transform_weights(self, matrix):
        with np.errstate(divide='ignore'):  # Ignore warnings for division by zero
            reciprocal_matrix = 1.0 / matrix
        reciprocal_matrix[np.isinf(reciprocal_matrix)] = 0  # Set infinities back to 0 (no connection)
        return reciprocal_matrix

    def compute_network_metrics(self, version='true', transform_weights=True):
        """Compute a variety of network metrics for true and predicted connectomes."""
        if version == 'true':
            matrix = self.true_connectome
        elif version == 'pred':
            matrix = self.pred_connectome

        # Create the graph
        G = nx.from_numpy_array(matrix)
        
        # Transform weights if specified
        # Do this because the weights must be seen as connection strength and a higher weight means e.g. shorter path length
        # if transform_weights:
        matrix_reciprocal = self.transform_weights(matrix)
        G_reciprocal = nx.from_numpy_array(matrix_reciprocal)

        # Use reciprocal weights for path-based metrics (shortest path length, efficiency, betweenness centrality).
        # Keep original weights for clustering, modularity, assortativity, and most centrality measures.

        # omega, sigma = self.small_worldness(G, 1)

        metrics = {
            # f'Node Strength {version}': dict(G.degree(weight='weight')),  # Weighted degree (strength)
            # f'Node Efficiency {version}': {node: self.node_efficiency(G, node) for node in G.nodes()}  # Efficiency for each node
            
            # Connectivity
            # f'Degree Centrality {version}': dict(G.degree(weight='weight')),  # Measures node importance based on number of connections, summing edge weights for each node (same as "node strength")
            # f'Eigenvector Centrality {version}': nx.eigenvector_centrality_numpy(G, weight='weight'),  # Measures influence of nodes based on connectivity to other central nodes.
            # f'Betweenness Centrality {version}': nx.betweenness_centrality(G, weight='weight', normalized=True),  # Measures frequency of a node on shortest paths, indicating its role in information flow.
            
            # Efficiency
            f'Global Efficiency {version}': nx.global_efficiency(G_reciprocal),  # Measures overall efficiency of information transfer in the network.
            f'Local Efficiency {version}': nx.local_efficiency(G_reciprocal),  # Measures fault tolerance, showing how well neighbors are connected if a node is removed.
            
            # Assortativity and Modularity
            f'Assortativity {version}': nx.degree_assortativity_coefficient(G, weight='weight'),  # Measures tendency of nodes to connect with similar-degree nodes.
            f'Modularity {version}': nx.algorithms.community.modularity(G, nx.algorithms.community.greedy_modularity_communities(G, weight='weight')),  # Measures strength of community structure.

            # Clustering and Centrality
            f'Clustering Coefficient {version}': nx.average_clustering(G, weight='weight'),  # Measures tendency of nodes to form tightly connected groups.
            f'Global Reaching Centrality {version}': self.global_reaching_centrality(G),  # Quantifies network's hierarchical structure based on how central nodes reach others.
            
            # Path and Distance
            f'Path Length {version}': nx.average_shortest_path_length(G_reciprocal, weight='weight'),  # Measures integration, as average shortest path between nodes. (same as "characteristic path length")

            # Small-worldness and Density
            # f'Small-worldness Omega {version}': omega,  # Quantifies small-world properties by comparing clustering and path length to random networks.
            # f'Small-worldness Sigma {version}': sigma,  # Alternative measure for small-worldness, combining clustering and path length.
            f'Network Density {version}': nx.density(G),  # Measures overall connectivity density, calculated as ratio of actual to possible edges.
        }


        self.results.update(metrics)
        
    def format_metrics(self):
        return """
            Metrics Summary:
            ----------------
            Metrics including all labels
            Accuracy: {Accuracy:.4f}
            F1-Score (Macro): {F1-Score (macro):.4f}
            Precision (Macro): {Precision (macro):.4f}
            Recall (Macro): {Recall (macro):.4f}
            F1-Score (Weighted): {F1-Score (weighted):.4f}
            Precision (Weighted): {Precision (weighted):.4f}
            Recall (Weighted): {Recall (weighted):.4f}
            
            Metrics ignoring the unknown (and potentially thresholded) labels
            Accuracy: {Accuracy without unknown:.4f}
            F1-Score (Macro): {F1-Score (macro) without unknown:.4f}
            Precision (Macro): {Precision (macro) without unknown:.4f}
            Recall (Macro): {Recall (macro) without unknown:.4f}
            F1-Score (Weighted): {F1-Score (weighted) without unknown:.4f}
            Precision (Weighted): {Precision (weighted) without unknown:.4f}
            Recall (Weighted): {Recall (weighted) without unknown:.4f}
            
            MSE: {MSE:.4f}
            Pearson Correlation: {Pearson Correlation:.4f}
            Spearman Correlation: {Spearman Correlation:.4f}
            Frobenius Norm: {Frobenius Norm:.4f}
            Earth Mover's Distance: {Earth Mover's Distance:.4f}

            """.format(**self.results)