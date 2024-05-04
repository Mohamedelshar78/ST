import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import community
from networkx.algorithms.community.quality import modularity
from community import community_louvain
from networkx.algorithms.community.centrality import girvan_newman
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


def PageRank(G):
    # Calculate PageRank scores
    pagerank_scores = nx.pagerank(G)
    print(pagerank_scores)
    root = tk.Tk()
    root.title("PageRank")
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20, pady=20, fill='both', expand=True)
    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20, pady=20)
    search_label = ttk.Label(left_frame, text="Filter by PageRank score ")
    search_label.grid(row=1, column=0)
    search_entry = ttk.Entry(left_frame)
    search_entry.grid(row=1, column=1)
    search_button = ttk.Button(left_frame, text="Filter")
    search_button.grid(row=1, column=2)
    # Create the table
    tree = ttk.Treeview(left_frame, columns=("Node", "PageRank Score"))
    tree.heading("Node", text="Node")
    tree.heading("PageRank Score", text="PageRank Score")
    #add canvas to display graph
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 10))
    node_size = [pagerank_scores[node] * 1000 for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
    plt.title("Graph")
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    quit_button = tk.Button(root, text="Quit", command=lambda: root.destroy())
    quit_button.pack(side=tk.BOTTOM)
    # Add the data to the table
    for node, score in pagerank_scores.items():
        tree.insert("", "end", text="", values=(node, round(score, 3)))
    # Add the table to the window
    tree.grid(row=2, column=0, columnspan=3)

    def filter_nodes():
        # Clear the previous selection
        tree.delete(*tree.get_children())
        # Get the filter threshold
        num = float(search_entry.get())
        node_colors = ['skyblue' if pagerank_scores[node] >= num else 'lightgray' for node in G.nodes()]
        # Filter the nodes based on the threshold
        for node, score in pagerank_scores.items():
            if score >= num:
                tree.insert("", "end", text="", values=(node, round(score, 3)))

        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors,node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
        plt.title("Graph")
        canvas.draw()

    # Bind the search button to the filter_nodes function
    search_button.config(command=filter_nodes)
    # Start the Tkinter event loop
    root.mainloop()
#===========================================================================================
def Degree_Centrality(G):
    # Calculate degree centrality
    dc = nx.degree_centrality(G)
    node_degrees = dict(G.degree())
    print(dc)

    root = tk.Tk()
    root.title("Degree Centrality")

    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20, pady=20, fill='both', expand=True)

    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20, pady=20)

    search_label = ttk.Label(left_frame, text="Filter by Degree centrality ")
    search_label.grid(row=1, column=0)
    search_entry = ttk.Entry(left_frame)
    search_entry.grid(row=1, column=1)
    search_button = ttk.Button(left_frame, text="Filter")
    search_button.grid(row=1, column=2)

    # Create the table
    tree = ttk.Treeview(left_frame, columns=("Node", "Degree","Degree Centrality"))
    tree.heading("Node", text="Node")
    tree.heading("Degree", text="Degree")
    tree.heading("Degree Centrality", text="Degree Centrality")

    #add canvas to display graph
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 10))

    node_size = [dc[node] * 1000 for node in G.nodes()]

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
    plt.title("Graph")

    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    quit_button = tk.Button(root, text="Quit", command=lambda: root.destroy())
    quit_button.pack(side=tk.BOTTOM)
    # Add the data to the table
    for node, degree_dict in dc.items():
        tree.insert("", "end", text="", values=(node, node_degrees[node], round(degree_dict, 3)))

    # Add the table to the window
    tree.grid(row=2, column=0, columnspan=3)

    def filter_nodes():
        # Clear the previous selection
        tree.delete(*tree.get_children())

        # Get the filter threshold
        num = float(search_entry.get())
        node_colors = ['skyblue' if dc[node] >= num else 'lightgray' for node in G.nodes()]

        # Filter the nodes based on the threshold
        for node, degree in dc.items():
            if degree >= num:
                tree.insert("", "end", text="", values=(node,node_degrees[node], round(degree, 3)))

        #H = G.subgraph([n for n in G.nodes() if dc[n] >= num])
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
        plt.title("Graph")
        canvas.draw()
    # Bind the search button to the filter_nodes function
    search_button.config(command=filter_nodes)
    # Start the Tkinter event loop
    root.mainloop()
#===========================================================================================
def Closeness_Centrality(G):
    # Calculate closeness centrality
    cc = nx.closeness_centrality(G)
    # Create the Tkinter window
    root = tk.Tk()
    root.title("Closeness Centrality")
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20, pady=20, fill='both', expand=True)  # Adjust the weight parameter here
    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20, pady=20)
    search_label = ttk.Label(left_frame, text="Filter by Closeness centrality ")
    search_label.grid(row=1, column=0)
    search_entry = ttk.Entry(left_frame)
    search_entry.grid(row=1, column=1)
    search_button = ttk.Button(left_frame, text="Filter")
    search_button.grid(row=1, column=2)

    # Create the table
    tree = ttk.Treeview(left_frame, columns=("Node",  "Closeness Centrality"))
    tree.heading("Node", text="Node")
    tree.heading("Closeness Centrality", text="Closeness Centrality")

    node_size = [cc[node] * 1000 for node in G.nodes()]
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
    plt.title("Graph")

    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Add the data to the table
    for node, degree in cc.items():
        tree.insert("", "end", text="", values=(node, round(degree, 3)))
    # Add the table to the window
    tree.grid(row=2, column=0, columnspan=3)

    def filter_nodes():
        # Clear the previous selection
        tree.delete(*tree.get_children())
        # Get the filter threshold
        num = float(search_entry.get())
        node_colors = ['skyblue' if cc[node] >= num else 'lightgray' for node in G.nodes()]
        # Filter the nodes based on the threshold
        for node, degree in cc.items():
            if degree >= num:
                tree.insert("", "end", text="", values=(node, round(degree, 3)))

        # Filter and draw the subgraph
        #H = G.subgraph([n for n in G.nodes() if cc[n] >= num])
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
        plt.title("Graph")
        canvas.draw()
    search_button.config(command=filter_nodes)
    # Start the Tkinter event loop
    root.mainloop()
#===========================================================================================
def Betweenness_Centrality(G):
    # Calculate betweenness centrality
    bc = nx.betweenness_centrality(G)
    # Create the Tkinter window
    root = tk.Tk()
    root.title("Betweenness Centrality")
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20, pady=20, fill='both', expand=True)  # Adjust the weight parameter here
    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20, pady=20)
    search_label = ttk.Label(left_frame, text="Filter by Betweenness centrality ")
    search_label.grid(row=1, column=0)
    search_entry = ttk.Entry(left_frame)
    search_entry.grid(row=1, column=1)
    search_button = ttk.Button(left_frame, text="Filter")
    search_button.grid(row=1, column=2)
    # Create the table
    tree = ttk.Treeview(left_frame, columns=("Node", "Source", "Betweenness Centrality"))
    tree.heading("Node", text="Node")
    tree.heading("Source", text="Source")
    tree.heading("Betweenness Centrality", text="Betweenness Centrality")
    # Add the table to the window
    tree.grid(row=2, column=0, columnspan=3)
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 10))
    node_size = [bc[node] * 1000 for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
    plt.title("Graph")

    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    # Add the data to the table
    for source, target in G.edges():
        betweenness_source = bc[source]
        tree.insert("", "end", text="", values=(target, source, round(betweenness_source, 3)))

    def filter_nodes():
        # Clear the previous selection
        tree.delete(*tree.get_children())
        # Get the filter threshold
        num = float(search_entry.get())
        # Initialize colors for nodes
        node_colors = ['skyblue' if bc[node] >= num else 'lightgray' for node in G.nodes()]

        # Filter the nodes based on the threshold
        for source, target in G.edges():
            if bc[source] >= num:
                betweenness_source = bc[source]
                tree.insert("", index=0, text="", values=(target, source, round(betweenness_source, 3)))

        #H = G.subgraph([n for n in G.nodes() if bc[n] >= num])
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
        plt.title("Graph")
        canvas.draw()
    search_button.config(command=filter_nodes)
    # Start the Tkinter event loop
    root.mainloop()
#===========================================================================================
def compare_community_detection(G):
    print('start compare community detection')
    if G.number_of_edges() == 0:
        print("Error: Graph has no edges.")
        return
    if G.is_directed():
        G=nx.to_undirected(G)
    # Girvan Newman Algorithm
    girvan_newman_communities_generator = nx.algorithms.community.girvan_newman(G)
    girvan_newman_communities = next(girvan_newman_communities_generator)
    girvan_newman_modularity = nx.algorithms.community.quality.modularity(G, girvan_newman_communities)

    # Louvain Algorithm
    louvain_communities = community_louvain.best_partition(G)
    louvain_communities_sets = {}
    for node, community in louvain_communities.items():
        if community not in louvain_communities_sets:
            louvain_communities_sets[community] = set()
        louvain_communities_sets[community].add(node)
    louvain_modularity = modularity(G, list(louvain_communities_sets.values()))
    # Display results in GUI
    root = tk.Tk()
    root.title("Community Detection Comparison")

    # Create labels and display results
    label_girvan_newman = ttk.Label(root, text="Girvan Newman Algorithm",font=("Helvetica", 10))
    label_girvan_newman.grid(row=0, column=0, padx=5, pady=5, sticky="w")

    label_girvan_newman_num_communities = ttk.Label(root, font=("Helvetica", 12),text=f"Number of Communities: {len(girvan_newman_communities)}")
    label_girvan_newman_num_communities.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    label_girvan_newman_modularity = ttk.Label(root,font=("Helvetica", 10), text=f"Modularity: {girvan_newman_modularity}")
    label_girvan_newman_modularity.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    label_louvain = ttk.Label(root, text="Louvain Algorithm",font=("Helvetica", 10))
    label_louvain.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    label_louvain_num_communities = ttk.Label(root, font=("Helvetica", 10),text=f"Number of Communities: {len(louvain_communities_sets)}")
    label_louvain_num_communities.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    label_louvain_modularity = ttk.Label(root, font=("Helvetica", 10),text=f"Modularity: {louvain_modularity}")
    label_louvain_modularity.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    # Draw the graph
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray')
    ax.set_title("Graph")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    root.mainloop()
#===========================================================================================
def partition_graph(G, algorithm='louvain'):
    if nx.is_directed(G):
        G=nx.to_undirected(G)

    if G.number_of_edges() == 0:
        print("Error: Graph has no edges.")
        return
    clusters = get_cluster(G, algorithm)
    com=len(clusters)
    results = evaluate_clustering(G,algorithm)
    print(results)
    visualize_clusters_gui(G,clusters,com,results[0],results[1],results[2])
#===========================================================================================
def visualize_clusters(G, clusters):
    if nx.is_directed(G):
        G=nx.to_undirected(G)
    # Convert sets to lists
    clusters = [list(cluster) for cluster in clusters]
    node_to_cluster = {node: cluster_idx for cluster_idx, cluster in enumerate(clusters) for node in cluster}
    colors = [node_to_cluster[node] for node in G.nodes()]
    pos = nx.spring_layout(G)
    nx.draw(G, pos,with_labels=True ,node_color=colors,node_size=1000,font_size=10, edge_color='gray' ,cmap=plt.cm.tab20)
#===========================================================================================
def visualize_clusters_gui(G, clusters,num_communities,Modularity,NMI,ARI):
    if nx.is_directed(G):
        G=nx.to_undirected(G)
    root = tk.Tk()
    root.title("Graph Clustering Visualization")
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20, pady=20, fill='both')  # Adjust the weight parameter here

    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20, pady=20)

    space = tk.Label(left_frame,height=5)
    space.grid(row=1, column=0)

    number_Nodes = tk.Label(left_frame, text=f"Number Of Comunity  : {num_communities} ",width=40,height=1,font=("Helvetica", 12))
    number_Nodes.grid(row=1, column=0)

    number_Edges = tk.Label(left_frame, text=f"Modularity  : {Modularity} ",font=("Helvetica", 12))
    number_Edges.grid(row=3, column=0)

    number_Edges = tk.Label(left_frame, text=f"NMI  : {NMI} ",font=("Helvetica", 12))
    number_Edges.grid(row=4, column=0)

    number_Edges = tk.Label(left_frame, text=f"ARI  : {ARI} ",font=("Helvetica", 12))
    number_Edges.grid(row=5, column=0)

    fig, ax = plt.subplots()
    visualize_clusters(G, clusters)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    quit_button = tk.Button(root, text="Quit", command=lambda: root.destroy())
    quit_button.pack(side=tk.BOTTOM)

    root.mainloop()
#===========================================================================================
def evaluate_clustering(graph, algorithm='louvain'):

    if nx.is_directed(graph):
        graph=nx.to_undirected(graph)
    # Ground truth (if available)
    # Add ground truth communities as node attributes for evaluation (optional)
    ground_truth_communities = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0,
                                10: 0, 11: 0, 12: 0, 13: 0, 14: 1, 15: 1, 16: 0, 17: 0, 18: 1, 19: 0,
                                20: 1, 21: 0, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1,
                                30: 1, 31: 1, 32: 1, 33: 1}
    nx.set_node_attributes(graph, ground_truth_communities, 'communities')

    ground_truth_communitiess = nx.get_node_attributes(graph, 'communities')
    ground_truth_labels = list(ground_truth_communities.values()) if ground_truth_communitiess else None

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
        ari = adjusted_rand_score(list(partition.values()), ground_truth_labels)

    return modularity, nmi, ari
#===========================================================================================
def get_cluster(G,algorithm='louvain'):
    if G.is_directed():
        G=nx.to_undirected(G)
    if G.number_of_edges() == 0:
        print("Error: Graph has no edges.")
        return
    if algorithm == 'louvain':
        # Girvan Newman Algorithm
        communities = girvan_newman(G)
        node_groups = []
        for com in next(communities):
            node_groups.append(list(com))

    elif algorithm == 'girvan_newman':
        # Girvan Newman Algorithm
        communities = girvan_newman(G)
        node_groups = []
        for com in next(communities):
            node_groups.append(list(com))
    elif algorithm == 'label_propagation':
        node_groups = list(nx.algorithms.community.label_propagation_communities(G))

    else:
        raise ValueError("Unsupported algorithm. Supported algorithms are 'louvain', 'girvan_newman', and 'label_propagation'.")
    return node_groups
#===========================================================================================
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

    visualize_each_clusters(graph,clusters)
    print(clusters)
    return clusters
#===========================================================================================
def modularity_based_partitioning(graph):
    # Perform modularity maximization
    partition = nx.community.greedy_modularity_communities(graph)

    # Convert partition format to dictionary
    clusters = {}
    for i, com in enumerate(partition):
        clusters[i] = list(com)

    visualize_each_clusters(graph,clusters)
    print(clusters)
    return clusters
#===========================================================================================
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

    visualize_each_clusters(graph,clusters)
    return clusters
#===========================================================================================
def visualize_each_clusters(graph, clusters):
    num_clusters = len(clusters)
    root = tk.Tk()
    root.title("graph partithon")
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
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    root.mainloop()