"""Network metrics for connectome matrices."""

import numpy as np
import networkx as nx


def compute_network_metrics(matrix, compute_advanced=True, compute_community=False):
    """Compute network metrics for a weighted connectome matrix."""
    metrics = {}

    if matrix is None or matrix.size == 0:
        return metrics

    G = nx.from_numpy_array(matrix)

    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    metrics['num_connected_components'] = nx.number_connected_components(G)
    metrics['average_clustering'] = nx.average_clustering(G, weight='weight')

    if compute_advanced:
        largest_cc = max(nx.connected_components(G), key=len) if nx.number_connected_components(G) > 0 else set()
        G_connected = G.subgraph(largest_cc).copy() if len(largest_cc) > 1 else G

        if G_connected.number_of_nodes() > 1 and nx.is_connected(G_connected):
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_weights = np.where(matrix > 0, 1.0 / matrix, 0.0)
            G_weighted = nx.from_numpy_array(inv_weights)
            G_weighted_connected = G_weighted.subgraph(largest_cc).copy()
            if nx.is_connected(G_weighted_connected):
                metrics['average_path_length'] = nx.average_shortest_path_length(G_weighted_connected, weight='weight')
                metrics['global_efficiency'] = nx.global_efficiency(G_weighted_connected)
            else:
                metrics['average_path_length'] = np.nan
                metrics['global_efficiency'] = np.nan
        else:
            metrics['average_path_length'] = np.nan
            metrics['global_efficiency'] = np.nan

        metrics['local_efficiency'] = nx.local_efficiency(G)
    else:
        metrics['average_path_length'] = np.nan
        metrics['global_efficiency'] = np.nan
        metrics['local_efficiency'] = np.nan

    try:
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight')
    except Exception:
        metrics['degree_assortativity'] = np.nan

    if compute_community:
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
            metrics['modularity'] = nx.algorithms.community.modularity(G, communities, weight='weight')
            metrics['num_communities'] = len(communities)
        except Exception:
            metrics['modularity'] = np.nan
            metrics['num_communities'] = np.nan
    else:
        metrics['modularity'] = np.nan
        metrics['num_communities'] = np.nan

    return metrics
