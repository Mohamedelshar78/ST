import networkx as nx
import pandas as pd
import community
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from networkx.algorithms.community import modularity

# Step 1: Read the node attributes from the first CSV file
node_df = r"F:\FCIS_2024\8Semester\Social\Task\nodes.csv"

# Step 2: Read the links from the second CSV file
links_df = r"F:\FCIS_2024\8Semester\Social\Task\links.csv"

# Step 3: Create the graph and add nodes with attributes
G = nx.Graph()
def load_network(node_path, edges_path):
    print('loading network...')
    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edges_path)
    half_nodes_df = nodes_df.sample(frac=1, random_state=1)
    print(half_nodes_df.head(10))
    for index, row in half_nodes_df.iterrows():
        G.add_node(row['ID'], attr_dict=row.to_dict())
    for index, row in edges_df.iterrows():
        if row['Source'] in G.nodes and row['Target'] in G.nodes:
            G.add_edge(row['Source'], row['Target'], attr_dict=row.to_dict())

    return G

G= load_network(node_df,links_df)

# Step 2: Partitioning based on Gender or Class
def partition_graph(G, attribute='gender_or_class'):
    partitions = {}
    for node in G.nodes(data=True):
        attr_val = node[1]['attr_dict'][attribute]  # Change 'gender_or_class' if needed
        if attr_val not in partitions:
            partitions[attr_val] = [node[0]]
        else:
            partitions[attr_val].append(node[0])
    return partitions

# Step 3: Visualization
def visualize_partitions(G, partitions):
    pos = nx.spring_layout(G)
    colors = ['r', 'b', 'g', 'y', 'c', 'm']  # Add more colors if needed

    plt.figure(figsize=(10, 8))
    for i, (attr_val, nodes) in enumerate(partitions.items()):
        subgraph = G.subgraph(nodes)
        nx.draw(subgraph, pos, node_color=colors[i % len(colors)], with_labels=True, label=attr_val)

    plt.title('Graph Partitioning based on Gender or Class')
    plt.legend()
    plt.show()

def visualize_each_clusters(graph, clusters):
    root = tk.Tk()
    root.title("Graph Partition")

    # Create a frame to contain all clusters' plots
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas to scroll through clusters
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a scrollbar to navigate through clusters
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame inside the canvas to hold the clusters' plots
    clusters_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=clusters_frame, anchor=tk.NW)

    # Function to configure the canvas scroll region
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox(tk.ALL))

    canvas.bind('<Configure>', on_configure)

    # Loop through clusters and visualize each one
    for i, (cluster_id, nodes) in enumerate(clusters.items()):
        # Create a subgraph for the cluster
        subgraph = graph.subgraph(nodes)
        # Draw the subgraph with nodes colored by cluster
        pos = nx.spring_layout(subgraph)
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # You can add more colors if needed
        fig, ax = plt.subplots()
        nx.draw(subgraph, pos, ax=ax, node_color=colors[i % len(colors)], with_labels=True)
        ax.set_title(f"Cluster {cluster_id}")
        ax.axis('off')

        # Embed the Matplotlib figure in the clusters_frame
        canvas_fig = FigureCanvasTkAgg(fig, master=clusters_frame)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    root.mainloop()

def evaluate_clusters(clusters, graph, ground_truth=None):
    """
    Evaluates the clustering results using internal and external evaluation metrics.

    Args:
        clusters (list): List of clusters, where each cluster is a list of nodes.
        graph (networkx.Graph): The graph representing the network.
        ground_truth (list or None): Optional ground truth labels if available.

    Returns:
        tuple: A tuple containing the values of modularity, normalized mutual information (NMI),
               Rand Index (RI), and Adjusted Rand Index (ARI).
    """
    # Calculate Modularity
    modularity_value = modularity(graph, clusters)

    # Calculate NMI (if ground truth is available)
    nmi_value = None
    if ground_truth is not None:
        nmi_value = normalized_mutual_info_score(ground_truth, clusters)

    # Calculate RI and ARI (if ground truth is available)
    ri_value, ari_value = None, None
    if ground_truth is not None:
        # Convert cluster labels to a format compatible with the metrics
        cluster_labels = [0] * graph.number_of_nodes()
        for i, cluster in enumerate(clusters):
            for node in cluster:
                cluster_labels[node] = i
        ri_value = adjusted_rand_score(ground_truth, cluster_labels)
        ari_value = ri_value

    return modularity_value, nmi_value, ri_value, ari_value

p = partition_graph(G, 'Class')
print(p)
x = evaluate_clusters(p,G)
print(x)