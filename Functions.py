import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import networkx as nx
from networkx.algorithms.cuts import conductance
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import community
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community.quality import modularity
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
    nx.draw(G, pos, node_color=list(pagerank_scores.values()), with_labels=True,cmap=plt.cm.Reds, arrows=G.is_directed())

    #nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
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
    nx.draw(G, pos, node_color=list(dc.values()), with_labels=True,cmap=plt.cm.Reds, arrows=G.is_directed())

    #nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
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
        node_colors = ['red' if dc[node] >= num else 'lightgray' for node in G.nodes()]

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
    nx.draw(G, pos, node_color=list(cc.values()), with_labels=True,cmap=plt.cm.Reds, arrows=G.is_directed())

    #nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
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
        node_colors = ['red' if cc[node] >= num else 'lightgray' for node in G.nodes()]
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
    nx.draw(G, pos, node_color=list(bc.values()), with_labels=True, cmap=plt.cm.Blues, arrows=G.is_directed())
    #nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
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
        node_colors = ['red' if bc[node] >= num else 'lightgray' for node in G.nodes()]

        # Filter the nodes based on the threshold
        for source, target in G.edges():
            if bc[source] >= num:
                betweenness_source = bc[source]
                tree.insert("", index=0, text="", values=(target, source, round(betweenness_source, 3)))

        #H = G.subgraph([n for n in G.nodes() if bc[n] >= num])
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='gray', linewidths=1, font_size=10, ax=ax)
        plt.title("Graph")
        canvas.draw()
    search_button.config(command=filter_nodes)
    # Start the Tkinter event loop
    root.mainloop()

#===========================================================================================
def compare_community_detection(G,algorithm):
    # Initialize dictionary to store results
    results = {}

    if algorithm == 'Girvan_Newman':
        # Girvan Newman algorithm
        gn_communities = tuple(sorted(c) for c in next(girvan_newman(G)))
        gn_modularity = modularity(G, gn_communities)
        gn_conductance = [conductance(G, c) for c in gn_communities]
        results['Girvan Newman'] = {'Communities': gn_communities,
                                    'Modularity': gn_modularity,
                                    'Conductance': gn_conductance}

    elif algorithm == 'Greedy_Modularity':
        # Louvain algorithm
        louvain_communities = greedy_modularity_communities(G)
        louvain_modularity = modularity(G, louvain_communities)
        louvain_conductance = [conductance(G, c) for c in louvain_communities]
        results['Louvain'] = {'Communities': louvain_communities,
                              'Modularity': louvain_modularity,
                              'Conductance': louvain_conductance}
    else:
        # Label Propagation algorithm
        lpa_communities = list(asyn_lpa_communities(G))
        lpa_modularity = modularity(G, lpa_communities)
        lpa_conductance = [conductance(G, c) for c in lpa_communities]
        results['Label Propagation'] = {'Communities': lpa_communities,
                                        'Modularity': lpa_modularity,
                                        'Conductance': lpa_conductance}
    for algo, data in results.items():
        print(f"Algorithm: {algo}")
        print(f"Number of Communities: {len(data['Communities'])}")
        print(f"Modularity: {data['Modularity']}")
        print("Conductance of each cluster:")
        for i, conductance_val in enumerate(data['Conductance']):
            print(f"Cluster {i+1}: {conductance_val:.5f}")
        print()
        visualize_communities(G,data['Communities'],algorithm)

# Visualize the graph with community detection results
def visualize_communities(G, communities,tittel):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8),)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=comm, node_color=colors[i % len(colors)], node_size=200, alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.5)

    plt.title(tittel)
    plt.show()

def compare_community_detection_intable(G):
    # Initialize dictionary to store results
    results = {}

    # Girvan Newman algorithm
    gn_communities = tuple(sorted(c) for c in next(girvan_newman(G)))
    gn_modularity = modularity(G, gn_communities)
    gn_conductance = [conductance(G, c) for c in gn_communities]
    results['Girvan Newman'] = {'Communities': gn_communities,
                                'Modularity': gn_modularity,
                                'Conductance': gn_conductance}

    # Louvain algorithm
    louvain_communities = greedy_modularity_communities(G)
    louvain_modularity = modularity(G, louvain_communities)
    louvain_conductance = [conductance(G, c) for c in louvain_communities]
    results['Louvain'] = {'Communities': louvain_communities,
                          'Modularity': louvain_modularity,
                          'Conductance': louvain_conductance}

    return results

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
def get_cluster(G, algorithm='louvain'):
    if G.number_of_edges() == 0:
        print("Error: Graph has no edges.")
        return

    if algorithm == 'louvain':
        communities = community.best_partition(G)
    elif algorithm == 'girvan_newman':
        communities = girvan_newman(G)
    elif algorithm == 'label_propagation':
        communities = nx.algorithms.community.label_propagation_communities(G)
    else:
        raise ValueError("Unsupported algorithm. Supported algorithms are 'louvain', 'girvan_newman', and 'label_propagation'.")


    # Visualize the graph with node colors representing communities
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Positions for all nodes
    cmap = plt.cm.get_cmap('viridis', max(communities.values()) + 1)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=list(communities.values()), cmap=cmap)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title('Graph with Communities')
    plt.colorbar(label='Community')
    plt.axis('off')
    plt.show()

    return communities

def display_results_in_table(G):
    results = compare_community_detection_intable(G)
    root = tk.Tk()
    root.title("Community Detection Results")

    columns = ('Algorithm', 'Num Communities', 'Modularity', 'Conductance')

    tree = ttk.Treeview(root, columns=columns, show='headings')

    for col in columns:
        tree.heading(col, text=col)

    for algo, data in results.items():
        num_communities = len(data['Communities'])
        modularity_val = data['Modularity']
        conductance_vals = data['Conductance']
        for i, conductance_val in enumerate(conductance_vals):
            tree.insert('', 'end', values=(algo, num_communities, modularity_val, conductance_val))

    tree.pack()
    root.mainloop()






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

# Step 2: Partitioning based on Gender or Class
def partition_graph_based_criatera(G, attribute='gender_or_class'):
    partitions = {}
    for node in G.nodes(data=True):
        attr_val = node[1]['attr_dict'][attribute]  # Change 'gender_or_class' if needed
        if attr_val not in partitions:
            partitions[attr_val] = [node[0]]
        else:
            partitions[attr_val].append(node[0])
    return partitions

# Step 3: Visualization
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
