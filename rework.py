import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import pandas as pd

node_path = r"F:\FCIS_2024\8Semester\Social\Task\nodes.csv"
edges_path = r"F:\FCIS_2024\8Semester\Social\Task\links.csv"
def load_network(node_path, edges_path):
    print('loading network...')
    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edges_path)
    G= nx.Graph()
    half_nodes_df = nodes_df.sample(frac=0.8, random_state=1)
    print(half_nodes_df.head(10))
    for index, row in half_nodes_df.iterrows():
        G.add_node(row['ID'], attr_dict=row.to_dict())
    for index, row in edges_df.iterrows():
        if row['Source'] in G.nodes and row['Target'] in G.nodes:
            G.add_edge(row['Source'], row['Target'], attr_dict=row.to_dict())

    return G

def degree_based_partitioning(graph, num_clusters):
    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(graph)

    # Sort nodes based on degree centrality
    sorted_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)

    # Assign nodes to clusters based on degree centrality
    clusters = {}
    for i in range(num_clusters):
        clusters[i] = []

    for i, node in enumerate(sorted_nodes):
        clusters[i % num_clusters].append(node)

    return clusters

def modularity_based_partitioning(graph):
    # Perform modularity maximization
    partition = nx.community.greedy_modularity_communities(graph)
    # Convert partition format to dictionary
    clusters = {}
    for i, com in enumerate(partition):
        clusters[i] = list(com)

    return clusters

def spectral_clustering(graph, num_clusters):
    # Step 1: Construct the Laplacian matrix
    laplacian_matrix = nx.laplacian_matrix(graph).toarray()
    # Step 2: Compute the eigenvectors corresponding to the smallest eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # Step 3: Use the smallest eigenvectors to embed the vertices into a lower-dimensional space
    embedding = sorted_eigenvectors[:, :num_clusters]
    # Step 4: Apply K-means clustering to the embedded vertices
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embedding)
    # Get the cluster assignments
    cluster_assignments = kmeans.labels_
    # Convert cluster assignments to dictionary format
    clusters = {}
    for i, label in enumerate(cluster_assignments):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters

# Example usage
# Create a random graph
G = load_network(node_path, edges_path)

# Degree-based partitioning
num_clusters = 2
degree_clusters = degree_based_partitioning(G, num_clusters)
print("Degree-based partitioning:", degree_clusters)

# Modularity-based partitioning
modularity_clusters = modularity_based_partitioning(G)
print("Modularity-based partitioning:", modularity_clusters)

# Spectral clustering
spectral_clusters = spectral_clustering(G, num_clusters)
print("Spectral clustering:", spectral_clusters)

def visualize_clusters(graph, clusters, title):
    num_clusters = len(clusters)
    root = tk.Tk()
    root.title(title)

    for i, (cluster_id, nodes) in enumerate(clusters.items()):
        # Create a subgraph for the cluster
        subgraph = graph.subgraph(nodes)
        # Draw the subgraph with nodes colored by cluster
        pos = nx.spring_layout(subgraph)
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots()
        nx.draw(subgraph, pos, ax=ax, node_color=colors[cluster_id], with_labels=True)
        ax.set_title(f"Cluster {cluster_id}")
        ax.axis('off')
        # Embed the Matplotlib figure in a Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    root.mainloop()

# Visualize degree-based partitioning
visualize_clusters(G, degree_clusters, "Degree-based Partitioning")

# Visualize modularity-based partitioning
visualize_clusters(G, modularity_clusters, "Modularity-based Partitioning")

# Visualize spectral clustering
visualize_clusters(G, spectral_clusters, "Spectral Clustering")
