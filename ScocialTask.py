import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from tkinter import filedialog
import Functions

G = nx.DiGraph()
pos = nx.random_layout(G)
Matrix = list
layoutTrree = 1

def load_network(node_path, edges_path):
    print('loading network...')
    nodes_df = pd.read_csv(node_path)
    edges_df = pd.read_csv(edges_path)
    global half_nodes_df
    half_nodes_df = nodes_df.sample(frac=1, random_state=1)
    print(half_nodes_df.head(10))
    for index, row in half_nodes_df.iterrows():
        G.add_node(row['ID'], attr_dict=row.to_dict())
    for index, row in edges_df.iterrows():
        if row['Source'] in G.nodes and row['Target'] in G.nodes:
            G.add_edge(row['Source'], row['Target'], attr_dict=row.to_dict())

    return G

# Function to update node colors and labels based on selected column and class
def calculate_metrics(graph):

    if nx.is_empty(graph):
        return [0, 0, 0, 0]  # Return a list with four zeros if the graph is empty

    print('Number of nodes:', nx.number_of_nodes(graph))
    Num_node = nx.number_of_nodes(graph)

    print('Number of edges:', nx.number_of_edges(graph))
    Num_edge = nx.number_of_edges(graph)

    degree_sequence = [d for n, d in graph.degree]
    degree_count = nx.degree_histogram(graph)

    # Calculate clustering coefficient
    clustering_coefficient = nx.average_clustering(graph)

    # Average path length
    try:
        average_path_length = nx.average_shortest_path_length(graph)
    except nx.NetworkXError:
        print("Graph is not connected. Cannot compute average shortest path length.")
        average_path_length = None

    print('Average path length:', average_path_length)
    print('Clustering Coefficients:', clustering_coefficient)

    return Num_edge, Num_node, clustering_coefficient, average_path_length

def calculate_node_size_degrees(G):
    print("Calculating node degrees...")
    node_counts = {}
    for edge in G.edges():
        for node in edge:
            if node in node_counts:
                node_counts[node] += 1
            else:
                node_counts[node] = 1

    # Set node sizes based on counts
    node_sizes = {node: 1000 + count * 200 for node, count in node_counts.items()}
    print("Node sizes:", node_sizes)

    nx.set_node_attributes(G, node_sizes, name='node_size')

def create_main_window():
    def radio_click1():
        global G
        G = nx.Graph()
    def radio_click2():
        global G
        G = nx.DiGraph()

    root = tk.Tk()
    root.title("Graph Viewer")
    left_frame = tk.Frame(root )
    left_frame.pack(side="left",fill='both')
    right_frame = tk.Frame(root)
    right_frame.pack(side="right",fill='both')

    #add canvas to display graph
    global pos
    global G
    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='gray', linewidths=1, font_size=10, ax=ax)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    tk.Radiobutton(left_frame, text="Undirected", value="1", command=radio_click1,font=("Helvetica", 10),height=2,highlightbackground='blue').grid(column=1, row=0)
    tk.Radiobutton(left_frame, text="Directed", value="2", command=radio_click2,font=("Helvetica", 10),height=2).grid(column=0, row=0)

    node_label = tk.Label(left_frame, text="Nodes CSV : ",font=("Helvetica", 12))
    node_label.grid(row=1, column=0)
    node_entry = tk.Entry(left_frame)
    node_entry.grid(row=1, column=1)
    browse_nodes_button = tk.Button(left_frame, text="Browse")
    browse_nodes_button.grid(row=1, column=2)

    edges_label = tk.Label(left_frame, text="Edges CSV :",font=("Helvetica", 12))
    edges_label.grid(row=2, column=0)
    edges_entry = tk.Entry(left_frame)
    edges_entry.grid(row=2, column=1)
    browse_edges_button = tk.Button(left_frame, text="Browse")
    browse_edges_button.grid(row=2, column=2)

    load_button = tk.Button(left_frame, text="Load Network",font=("Helvetica", 12))
    load_button.grid(row=3, column=0, columnspan=3)

    columns_var = tk.StringVar(left_frame)
    columns_var.set("None")
    columns_label = tk.Label(left_frame, text="Select Node Column:")
    columns_label.grid(row=4, column=0)
    columns_menu = tk.OptionMenu(left_frame, columns_var, "None")
    columns_menu.grid(row=4, column=1)

    show_id_var = tk.BooleanVar(left_frame)
    show_id_var.set(False)
    show_id_checkbox = tk.Checkbutton(left_frame, text="Show Node label", variable=show_id_var)
    show_id_checkbox.grid(row=5, column=0)

    node_size_label = tk.Label(left_frame, text="Node Size :")
    node_size_label.grid(row=6, column=0)

    node_size_var = tk.IntVar(left_frame)
    node_size_var.set(100)
    node_size_spinbox = tk.Spinbox(left_frame, from_=1, to=1000000, increment=100, textvariable=node_size_var)
    node_size_spinbox.grid(row=6, column=1)

    edge_color_var = tk.StringVar(left_frame)
    edge_color_var.set("gray")
    edge_color_label = tk.Label(left_frame, text="Select Edge Color :")
    edge_color_label.grid(row=7, column=0)
    edge_color_menu = tk.OptionMenu(left_frame, edge_color_var,
                                    *['gray', 'gray','Blue','Red','Green', 'Yellow', 'Orange', 'Purple', 'Cyan'])
    edge_color_menu.grid(row=7, column=1)

    node_color_var = tk.StringVar(left_frame)
    node_color_var.set("skyblue")
    node_color_label = tk.Label(left_frame, text="Select Node Color :")
    node_color_label.grid(row=8, column=0)
    node_color_menu = tk.OptionMenu(left_frame, node_color_var,
                                    *['skyblue','Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Cyan'])
    node_color_menu.grid(row=8, column=1)

    edge_size_label = tk.Label(left_frame, text="Edge Size : ")
    edge_size_label.grid(row=9, column=0)

    edge_size_var = tk.IntVar(left_frame)
    edge_size_var.set(1)
    edge_size_spinbox = tk.Spinbox(left_frame, from_=0.5, to=10, increment=0.1, textvariable=edge_size_var)
    edge_size_spinbox.grid(row=9, column=1)

    calculate_degrees_button = tk.Label(left_frame, text="Change Edge Size ")
    calculate_degrees_button.grid(row=9, column=0)

    space = tk.Label(left_frame)
    space.grid(row=10, column=0)

    submit_button = tk.Button(left_frame, text="Submit Changes ",bd=5,bg="Green",font=("Helvetica", 9))
    submit_button.grid(row=11)



    fr_layout_button = tk.Button(left_frame, text=" Fruchterman La/yout")
    fr_layout_button.grid(row=12, column=1)

    random_layout_button = tk.Button(left_frame, text="Random Layout")
    random_layout_button.grid(row=12, column=3)

    tree_layout_button = tk.Button(left_frame, text="Tree Layout")
    tree_layout_button.grid(row=12, column=0)
    x = tk.Label(left_frame)
    x.grid(row=13, column=0)

    space = tk.Label(left_frame)
    space.grid(row=14, column=0)

    number_Nodes = tk.Label(left_frame, text="Number Of Nodes  : ",width=40,height=1,font=("Helvetica", 12))
    number_Nodes.grid(row=15, column=0)

    number_Edges = tk.Label(left_frame, text="Number Of Edges  :  ",font=("Helvetica", 12))
    number_Edges.grid(row=16, column=0)

    clustrig = tk.Label(left_frame, text="Clustring Coofficents  :",font=("Helvetica", 12))
    clustrig.grid(row=17, column=0)

    Avg = tk.Label(left_frame, text="Average path length  :",font=("Helvetica", 12))
    Avg.grid(row=18, column=0)

    space2 = tk.Label(left_frame,height=3)
    space2.grid(row=19, column=0)

    filter = tk.Label(left_frame, text="Filtaration  :",font=("Helvetica", 12))
    filter.grid(row=20, column=0)

    options = ["Degree centrality", "Closeness centrality",
               "Betweenness centrality",'PageRank']
    selected_option1 = tk.StringVar()
    combobox1 = ttk.Combobox(left_frame, values=options, textvariable=selected_option1)
    def combobox_selected(event):
        selected_algo = selected_option1.get()
        if selected_algo == options[0]:
            print("degree centrality")
            Functions.Degree_Centrality(G)
        elif selected_algo == options[1]:
            print("clossness centrality")
            Functions.Closeness_Centrality(G)
        elif selected_algo == options[2]:
            print('betweenness centrality')
            Functions.Betweenness_Centrality(G)
        elif selected_algo == options[3]:
            print('PageRank')
            Functions.PageRank(G)
    combobox1.bind("<<ComboboxSelected>>", combobox_selected)
    combobox1.grid(row=20, column=1)
    selected_option1.set(options[0])

    space3 = tk.Label(left_frame)
    space3.grid(row=21, column=0)

    #comunity partithon
    filter = tk.Label(left_frame, text="Compare  :",font=("Helvetica", 12))
    filter.grid(row=22, column=0)

    options2 = ["louvain", "girvan_newman",
               "label_propagation"]
    selected_option = tk.StringVar()
    combobox = ttk.Combobox(left_frame, values=options2, textvariable=selected_option)
    def combobox_selected2(event):
        selected_algo = selected_option.get()
        if selected_algo == options2[0]:
            Functions.partition_graph(G)
        elif selected_algo == options2[1]:
            Functions.partition_graph(G,'girvan_newman')
        elif selected_algo == options2[2]:
            Functions.partition_graph(G,'label_propagation')
    combobox.bind("<<ComboboxSelected>>", combobox_selected2)
    combobox.grid(row=22, column=1)
    selected_option.set(options2[0])


    num_of_patrtion = tk.Label(left_frame, text="Num Of Clusters :")
    num_of_patrtion.grid(row=23, column=0)

    num_of_patrtion_var = tk.IntVar(left_frame)
    num_of_patrtion_var.set(2)
    num_of_patrtion_spinbox = tk.Spinbox(left_frame, from_=2, to=10, increment=1, textvariable=num_of_patrtion_var)
    num_of_patrtion_spinbox.grid(row=23, column=1)



    # Graph Partitioning and Clustering partithon
    partithon = tk.Label(left_frame, text="partithon clusters :",font=("Helvetica", 12))
    partithon.grid(row=24, column=0)

    options2 = ["Degree-based partitioning", "Modularity-based partitioning",
                "Spectral clustering"]
    partithon_option = tk.StringVar()
    combobox = ttk.Combobox(left_frame, values=options2, textvariable=partithon_option)
    def combobox_selected2(event):
        selected_algo = partithon_option.get()
        num = num_of_patrtion_var.get()
        if selected_algo == options2[0]:
            Functions.degree_based_partitioning(G,num)
        elif selected_algo == options2[1]:
            Functions.modularity_based_partitioning(G)
        elif selected_algo == options2[2]:
            Functions.spectral_clustering(G,num)
    combobox.bind("<<ComboboxSelected>>", combobox_selected2)
    combobox.grid(row=24, column=1)
    partithon_option.set(options2[0])

    def load_network_wrapper():
        global Matrix
        load_network(node_entry.get(), edges_entry.get())
        update_node_colors_and_labels()
        update_dropdown_list()
        calculate_metrics_wrapper()

    def browse_nodes():
        node_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")],title="Select Nodes file")
        node_entry.delete(0, tk.END)
        node_entry.insert(0, node_path)

    def browse_edges():
        edges_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")],title="Select Edges file")
        edges_entry.delete(0, tk.END)
        edges_entry.insert(0, edges_path)

    def update_node_colors_and_labels():
        column_name = columns_var.get()
        show_id = show_id_var.get()

        # Check if any node in G.nodes is missing from the pos dictionary
        if any(node not in pos for node in G.nodes):
            print("Warning: Some nodes do not have positions assigned. Assigning positions using spring layout.")
            # Assign positions to missing nodes using spring layout
            if layoutTrree==2:
                new_positions = nx.drawing.fruchterman_reingold_layout(G)
                # Update the pos dictionary with the newly assigned positions for missing nodes
                pos.update(new_positions)
            elif layoutTrree==3:
                new_positions = nx.spring_layout(G)
                # Update the pos dictionary with the newly assigned positions for missing nodes
                pos.update(new_positions)
            else:
                new_positions = nx.random_layout(G)
                # Update the pos dictionary with the newly assigned positions for missing nodes
                pos.update(new_positions)

        # Update node colors based on selected column
        if column_name == "None":
            node_colors = [node_color_var.get() for _ in range(len(G.nodes))]
        else:
            unique_values = half_nodes_df[column_name].unique()
            node_color_map = {val: plt.cm.jet(random.random()) for val in unique_values}
            node_colors = [node_color_map[G.nodes[node]['attr_dict'][column_name]] for node in G.nodes()]
        nx.set_node_attributes(G, dict(zip(G.nodes, node_colors)), name='node_color')

        # Update node labels
        labels = {}
        for node in G.nodes():
            if show_id:
                labels[node] = str(node)  # Set node label to its ID
            else:
                labels[node] = ''  # Set empty label
        nx.set_node_attributes(G, labels, name='node_label')

        # Initialize node sizes dictionary with default values for all nodes
        node_sizes = {node: node_size_var.get() for node in G.nodes()}

        # Get node sizes from node_sizes attribute
        node_sizes_attr = nx.get_node_attributes(G, 'node_size')
        if node_sizes_attr:
            node_sizes.update(node_sizes_attr)

        edge_thickness = float(edge_size_var.get())
        ax.clear()
        nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors,
                node_size=[node_sizes[node] for node in G.nodes()],
                edge_color=edge_color_var.get(), font_size=10, width=edge_thickness, ax=ax)
        canvas.draw()
        print("Canvas updated")

    def submit_changes():
        update_node_colors_and_labels()

    def apply_fr_layout():
        global layoutTrree
        layoutTrree=2
        global pos
        pos=nx.drawing.layout.fruchterman_reingold_layout(G)
        update_node_colors_and_labels()

    def apply_tree_layout():
        global layoutTrree
        global pos
        pos=nx.drawing.spring_layout(G,dim=3)
        layoutTrree=3
        update_node_colors_and_labels()

    def apply_tree_layout():
        global layoutTrree
        global pos
        pos=nx.drawing.random_layout(G)
        layoutTrree=1
        update_node_colors_and_labels()

    def update_dropdown_list():
        menu = columns_menu["menu"]
        menu.delete(0, "end")  # Clear the existing options
        for column in half_nodes_df.columns[1:]:  # Start from the second column
            menu.add_command(label=column, command=tk._setit(columns_var, column))
    def calculate_metrics_wrapper():
        matrix = calculate_metrics(G)
        n_n=matrix[0]
        print("Number of nodes:", n_n)
        print(matrix[1])
        number_Nodes.config(text=f"Number Of Nodes  : {n_n}",font=("Helvetica", 8))
        number_Edges.config(text=f"Number Of Edges  :  {str(matrix[1])}",font=("Helvetica", 8))
        clustrig.config(text=f"Clustering Coefficients  : {matrix[2]}",font=("Helvetica", 8))
        Avg.config(text=f"Average path length  : {matrix[3]}",font=("Helvetica", 8))

    browse_nodes_button.config(command=lambda: browse_nodes())
    browse_edges_button.config(command=lambda: browse_edges())
    load_button.config(command=lambda: load_network_wrapper())
    show_id_checkbox.config(command=lambda: submit_changes())
    submit_button.config(command=lambda: submit_changes())
    fr_layout_button.config(command=lambda: apply_fr_layout())
    tree_layout_button.config(command=lambda: apply_tree_layout())
    random_layout_button.config(command=lambda: apply_tree_layout())
    root.mainloop()

# Call the function to create the main window
create_main_window()

