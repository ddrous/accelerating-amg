import dgl
import networkx as nx
from pyvis.network import Network


def visualize(graph, file_name='graph.html', nb_nodes_plot=50, notebook=False):

    if nb_nodes_plot is None:
        nb_nodes_plot = graph.number_of_nodes()
    
    assert nb_nodes_plot <= graph.number_of_nodes()
    
    g_dgl = graph.subgraph(list(range(nb_nodes_plot)))

    # Step 2. Convert the DGLGraph to a NetworkX graph
    g_netx = nx.Graph(g_dgl.to_networkx())

    # Step 3. Get and assign colors to networkX graph as node attributes 
    classes = g_dgl.ndata['label'].numpy()
    c_dict = {0:'red', 1:'green', 2:'blue'}
    colors = {i:c_dict.get(classes[i], 'black') for i in range(nb_nodes_plot)}
    nx.set_node_attributes(g_netx, colors, name="color")

    # Step 4. Get and assign sizes proportional to the classes found in DGL
    sizes = {i:int(classes[i])+1 for i in range(nb_nodes_plot)}
    nx.set_node_attributes(g_netx, sizes, name="size")

    # Step 5. Get and assign pyvis labels for elegant plotting
    labels = {i:str(i) for i in range(nb_nodes_plot)}
    nx.set_node_attributes(g_netx, labels, name="label")

    # Step 6. Remap the node ids to strings to avoid error with PyVis
    g_netx = nx.relabel_nodes(g_netx, labels)

    # Step 7. Plot the resulting netwrokX graph using PyVis
    g_pyvis = Network(height=1500, width=1500, notebook=notebook)
    g_pyvis.from_nx(g_netx, node_size_transf=lambda n:5*n)
    g_pyvis.show_buttons(filter_=['nodes'])
    g_pyvis.show(file_name)
