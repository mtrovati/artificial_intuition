
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import math
# Linux
#data_csv = pd.read_csv('/home/marcello/Dropbox/Artificial_Intuition_Research/Python_coding_validation/grouped_relations_all_csv_files.csv',names = ['source', 'target', 'edge'], header=None)
#data_conceptnetcsv = pd.read_csv('/home/marcello/Dropbox/Artificial_Intuition_Research/Python_coding_validation/data_conceptnet/grouped_conceptnet_dataset.csv', names = ['source', 'target', 'edge'], header=None)

# Windows Desktop
#data = pd.read_csv('D:\Dropbox\Artificial_Intuition_Research\Python_coding_validation\grouped_relations_all_csv_files.csv')

# IMac
data_csv = pd.read_csv('/Users/marcellotrovati/Dropbox/Artificial_Intuition_Research/Python_coding_validation/grouped_relations_all_csv_files.csv',names = ['source', 'target', 'edge'], header=None)
data_conceptnetcsv = pd.read_csv('/Users/marcellotrovati/Dropbox/Artificial_Intuition_Research/Python_coding_validation/data_conceptnet/grouped_conceptnet_dataset.csv')

kg_df = pd.DataFrame(data_csv, columns = ['source', 'target', 'edge'])

conceptnet_df = pd.DataFrame(data_conceptnetcsv, columns = ['source', 'target', 'edge'])

def draw_network(input_data_frame):
    cmap = plt.cm.get_cmap("Blues")
    # ALL EDGESG=nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    G=nx.from_pandas_edgelist(input_data_frame[input_data_frame['edge']==1], "source", "target", edge_attr=True)

    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='black', pos = pos)
    #plt.show()
    plt.savefig("Wikipedia_weight_1_network.png", format="PNG")
    
def net_analysys(input_data_frame):
    # ALL EDGESG=nx.from_pandas_edgelist(input_data_frame, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    G=nx.from_pandas_edgelist(input_data_frame, "source", "target", edge_attr=True)

    return nx.info(G), nx.density(G), G

def net_analysys_conceptnet(input_data_frame):
    # Create a graph with edges with weight = 1
    #G=nx.from_pandas_edgelist(input_data_frame[input_data_frame['edge']==1], "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    
    # Create a graph with all edges
    G=nx.from_pandas_edgelist(input_data_frame, "source", "target", edge_attr=True)

    return nx.info(G), nx.density(G), G

def compare_networks(G1, G2):

    nodes_G1 = G1.nodes
    nodes_G2 = G2.nodes

    nodes_in_G2_not_in_G1 = []
    nodes_in_G2_and_G1 = []

    for node in nodes_G2:
        if node not in nodes_G1:
            nodes_in_G2_not_in_G1.append(node)
        else:
            nodes_in_G2_and_G1.append(node)


    return nodes_in_G2_not_in_G1, nodes_in_G2_and_G1

def combine_networks(G1, G2):
    return nx.compose(G1,G2)


# Get all the nodes with a specific keyword in them

def get_nodes_with_keyword(keyword, G):
    nodes_with_keyword = []
    nodes = list(G.nodes)
    for node in nodes:
        if keyword in str(node):
            nodes_with_keyword.append(node)
    return nodes_with_keyword


# Get all the paths between two vertices v1 and v2 with cutoff = n in graph G
def get_all_paths_between_2_nodes(G, v1,v2,n):
    path_list = []
    for path in nx.all_simple_paths(G, v1, v2, cutoff=n):
        path_parsed = list(path)
        if path_parsed not in path_list:
            path_list.append(path_parsed)
    return path_list

# Get the edges (as node pairs) from a path
def get_edges_from_path(path_input):
    edge = []
    path = []
    for item in path_input:
        path = []
        for i in range(len(item)-1):
            path.append([item[i], item[i+1]])
        #print(path)
        edge.append(path)
    return edge


# Get the weights of a sequence of edges (described and node pairs) from an input path

def get_weights_from_path(input_path, G):
    #list_edges_with_weights = []
    edge_with_weight = []
    for edge in input_path:
        weight = G.get_edge_data(edge[0], edge[1])
        edge_with_weight.append([edge[0], edge[1], weight.get('edge')])

    return edge_with_weight

# Based on the above method, we shall now combine all the paths with weights between two nodes

def combine_paths_with_weights(input_paths,G):
    weighted_paths = []
    for path in input_paths:
        weighted_paths.append(get_weights_from_path(path,G))
    return weighted_paths



# Given two nodes and cutoff n, get all the paths and corresponding weights
def get_all_weighted_paths_between_two_nodes(keyword_1, keyword_2, cutoff, G):
    paths_between_2_nodes = get_all_paths_between_2_nodes(G, keyword_1, keyword_2, cutoff)
    return combine_paths_with_weights(get_edges_from_path(paths_between_2_nodes), G)


# Create the sub-network based on the paths between two keywords
def plot_subnet(list_paths):
    G_sub = nx.Graph()
    for item in list_paths:
        for subitem in item:
            G_sub.add_edge(subitem[0], subitem[1], weight = subitem[2])
    e_1 = [(u, v) for (u, v, d) in G_sub.edges(data=True) if d['weight'] == 1]
    e_2 = [(u, v) for (u, v, d) in G_sub.edges(data=True) if d['weight'] == 2]
    e_3 = [(u, v) for (u, v, d) in G_sub.edges(data=True) if d['weight'] == 3]
    e_4 = [(u, v) for (u, v, d) in G_sub.edges(data=True) if d['weight'] == 4]
    pos = nx.spring_layout(G_sub)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G_sub, pos, node_size=700)

    # edges 
    nx.draw_networkx_edges(G_sub, pos, edgelist=e_1,width=6, alpha=0.5, edge_color='b')
    nx.draw_networkx_edges(G_sub, pos, edgelist=e_2,width=6, alpha=0.5, edge_color='r')
    nx.draw_networkx_edges(G_sub, pos, edgelist=e_3,width=6, alpha=0.5, edge_color='g')
    nx.draw_networkx_edges(G_sub, pos, edgelist=e_4,width=6, alpha=0.5, edge_color='k')

    # labels
    nx.draw_networkx_labels(G_sub, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show()
    return G_sub

# Create and differentiate two sub-networks (from Wikipedia and ConceptNet) based on the paths between two keywords
def subnets_wiki_conceptnet(list_paths_wiki, list_paths_conceptnet):
    G = nx.Graph()
    for item in list_paths_wiki:
        for subitem in item:
            G.add_edge(subitem[0], subitem[1], weight = subitem[2])
    e_1 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 1]
    e_2 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 2]
    e_3 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 3]
    e_4 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 4]
    nodes_wiki = []
    for node in G.nodes():
        nodes_wiki.append(node)
    print(G.nodes())


    for item in list_paths_conceptnet:
        for subitem in item:
            G.add_edge(subitem[0], subitem[1], weight = subitem[2])
    e_c_1 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 1]
    e_c_2 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 2]
    e_c_3 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 3]
    e_c_4 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 4]
    
    nodes_conceptnet = []
    for node in G.nodes():
        if node not in nodes_wiki:
            nodes_conceptnet.append(node)
    print(G.nodes())
    #print(len(nodes_conceptnet))
    #print(len(nodes_wiki))

    pos = nx.spring_layout(G)  # positions for all nodes
     

    # nodes
    nx.draw_networkx_nodes(G, pos, nodelist= nodes_wiki, node_color = 'blue',  node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist= nodes_conceptnet, node_color = 'red', node_size=700)

    # # edges 
    nx.draw_networkx_edges(G, pos, edgelist=e_1,width=6, alpha=0.5, edge_color='b')
    nx.draw_networkx_edges(G, pos, edgelist=e_2,width=6, alpha=0.5, edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=e_3,width=6, alpha=0.5, edge_color='g')
    nx.draw_networkx_edges(G, pos, edgelist=e_4,width=6, alpha=0.5, edge_color='k')


    nx.draw_networkx_edges(G, pos, edgelist=e_c_1,width=6, alpha=0.5, edge_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=e_c_2,width=6, alpha=0.5, edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=e_c_3,width=6, alpha=0.5, edge_color='g')
    nx.draw_networkx_edges(G, pos, edgelist=e_c_4,width=6, alpha=0.5, edge_color='k')

    # # labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show() 
    return G



# Define the mathematical model to assess the paths
# At the moment, this will only be a dummy model and it will return the list of all the weights of the edges in each path
def list_of_weights_from_paths(list_input_paths):
    list_weights = []
    for path in list_input_paths:
        temp = []
        for edge in path:
            temp.append(edge[2])
        list_weights.append(temp)

    return list_weights


def information_propagation_between_2_nodes(prob_node_input, weight_edge):
    # note the influence on a node is the product of the information propagation from all its neighbours
    return prob_node_input*math.exp(weight_edge +(1-weight_edge))


#print('Network information' + "\n" + net_analysys(kg_df)[0])
#print('Network Density: ' + str(net_analysys(kg_df)[1]))

#print('-------------------')

#print('Network information' + "\n" + net_analysys_conceptnet(data_conceptnetcsv)[0])
#print('Network Density: ' + str(net_analysys_conceptnet(data_conceptnetcsv)[1]))

#print(list(net_analysys(kg_df)[2].nodes))
#print(list(net_analysys_conceptnet(data_conceptnetcsv)[2].nodes))

#print(len(compare_networks(net_analysys_conceptnet(data_conceptnetcsv)[2], net_analysys(kg_df)[2])[1]))

#G = combine_networks(net_analysys_conceptnet(data_conceptnetcsv)[2], net_analysys(kg_df)[2])
#for path in nx.all_simple_paths(G, source='temperature', target='rain'):
#    print(path)


if __name__ == "__main__":
    G1 = net_analysys(kg_df)[2]
    G2 = net_analysys_conceptnet(conceptnet_df)[2]
    G_wiki_conceptnet = combine_networks(G1, G2)

    #print(net_analysys_conceptnet(conceptnet_df)[0])
    #print(net_analysys(kg_df)[0])
    #print(nx.info( G_wiki_conceptnet))
    #print(nx.is_directed(G1))

    #print(G1.has_node('severe weather weather'))
    #print(G2.has_node('weather')) 
    A = get_all_weighted_paths_between_two_nodes('weather forecasts', 'aircraft', 6, G1)
    B = get_all_weighted_paths_between_two_nodes('weather forecasts', 'aircraft', 6, G_wiki_conceptnet)
    print(A)
    print("==============")
    print(B)
    #print(mathematical_model_from_list_of_paths(A))
   # draw_network(kg_df)
    #subnets_wiki_conceptnet(A,B)
   

    #plot_subnet(A)
