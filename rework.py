import networkx as nx
import community
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt


def evaluate_clustering(graph, algorithm='louvain'):
    """
    Evaluate the quality of a graph clustering using modularity, NMI, and ARI.

    Parameters:
        graph (NetworkX graph): The input graph.
        algorithm (str): The clustering algorithm to use. Supported values are 'louvain',
                         'girvan_newman', and 'label_propagation'. Defaults to 'louvain'.

    Returns:
        tuple: A tuple containing modularity, NMI, and ARI scores.
    """
    # Ground truth (if available)
    ground_truth_communities = nx.get_node_attributes(graph, 'communities')
    ground_truth_labels = list(ground_truth_communities.values()) if ground_truth_communities else None

    # Detect communities
    if algorithm == 'louvain':
        partition = community.best_partition(graph)
    elif algorithm == 'girvan_newman':
        clusters_generator = nx.algorithms.community.girvan_newman(graph)
        partition = {node: idx for idx, cluster in enumerate(next(clusters_generator)) for node in cluster}
    elif algorithm == 'label_propagation':
        partition = {node: idx for idx, cluster in enumerate(nx.algorithms.community.label_propagation_communities(graph)) for node in cluster}
    else:
        raise ValueError("Unsupported algorithm. Supported algorithms are 'louvain', 'girvan_newman', and 'label_propagation'.")

    # Modularity (internal evaluation)
    modularity = community.modularity(partition, graph)

    # External evaluation metrics
    nmi, ari = None, None
    if ground_truth_labels:
        # NMI (Normalized Mutual Information)
        nmi = normalized_mutual_info_score(list(partition.values()), ground_truth_labels)

        # ARI (Adjusted Rand Index)
        ari = adjusted_rand_score(list(partition.values()), list(partition.values()))

    return modularity, nmi, ari

def clustering_evaluation(graph, algorithms=['louvain', 'girvan_newman', 'label_propagation']):
    """
    Perform clustering evaluation using multiple algorithms and metrics.

    Parameters:
        graph (NetworkX graph): The input graph.
        algorithms (list): List of clustering algorithms to evaluate. Defaults to
                           ['louvain', 'girvan_newman', 'label_propagation'].

    Returns:
        dict: A dictionary containing evaluation results for each algorithm.
    """
    evaluation_results = {}

    for algorithm in algorithms:
        modularity, nmi, ari = evaluate_clustering(graph, algorithm)
        evaluation_results[algorithm] = {'modularity': modularity, 'NMI': nmi, 'ARI': ari}

    return evaluation_results

# Example usage:
# Assuming you have a NetworkX graph named 'graph'
# Create a sample graph (you can replace this with your own graph)
graph = nx.karate_club_graph()

# Add ground truth communities as node attributes for evaluation (optional)
ground_truth_communities = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0,
                            10: 0, 11: 0, 12: 0, 13: 0, 14: 1, 15: 1, 16: 0, 17: 0, 18: 1, 19: 0,
                            20: 1, 21: 0, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
                            30: 1, 31: 1, 32: 1, 33: 1}
nx.set_node_attributes(graph, ground_truth_communities, 'communities')

# Perform clustering evaluation
evaluation_results = clustering_evaluation(graph)

# Print evaluation results
for algorithm, results in evaluation_results.items():
    print(f"Algorithm: {algorithm}")
    print(f"Modularity: {results['modularity']}")
    print(f"NMI: {results['NMI']}")
    print(f"ARI: {results['ARI']}")
    print()

# Optionally, you can visualize the graph with community colors
pos = nx.spring_layout(graph)
node_colors = [ground_truth_communities[node] for node in graph.nodes()]
nx.draw(graph, pos, node_color=node_colors, with_labels=True, cmap=plt.cm.tab10)
plt.show()
