import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.cuts import conductance

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
        louvain_communities = nx.community.louvain_communities(G)
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
            print(f"Cluster {i+1}: {conductance_val:.4f}")
        print()

# Visualize the graph with community detection results
def visualize_communities(G, communities,tittel):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8),)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=comm, node_color=colors[i % len(colors)], node_size=200, alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.title(tittel)
    plt.show()

# Example usage
G = nx.karate_club_graph()
results = compare_community_detection(G,'Greedy_Modularity')
