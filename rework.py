import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd

nodes_file = r"F:\FCIS_2024\8Semester\Social\Task\nodes.csv"
edges_file = r"F:\FCIS_2024\8Semester\Social\Task\links.csv"

def load_network(node_path, edges_path):
    print('loading network...')
    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edges_path)
    half_nodes_df = nodes_df.sample(frac=0.1, random_state=1)
    G = nx.Graph()
    print(half_nodes_df.head(10))
    for index, row in half_nodes_df.iterrows():
        G.add_node(row['ID'], attr_dict=row.to_dict())
    for index, row in edges_df.iterrows():
        if row['Source'] in G.nodes and row['Target'] in G.nodes:
            G.add_edge(row['Source'], row['Target'], attr_dict=row.to_dict())

    return G
G = load_network(nodes_file, edges_file)

# Calculate betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)
# Visualize the graph with node colors based on betweenness centrality
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=list(betweenness_centrality.values()), with_labels=True, cmap=plt.cm.Blues)
plt.title('Network Visualization with Node Color Based on Betweenness Centrality')
plt.show()

# Calculate PageRank
page_rank = nx.pagerank(G, alpha=0.85)
# Visualize the graph with node colors based on PageRank
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=list(page_rank.values()), with_labels=True, cmap=plt.cm.Reds)
plt.title('Network Visualization with Node Color Based on PageRank')
plt.show()