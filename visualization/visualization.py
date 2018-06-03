import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from neat.genome import Genome


def plot_genome(genome: Genome):
    G = nx.DiGraph()
    sensor_list = []
    output_list = []
    rest = []
    for node_nr, node in enumerate(genome.nodes):
        color_map = []
        if node[1] == 'sensor':
            color_map.append(0)
            sensor_list.append(node[0])
            attr_dict = {'pos': (00, 0.1)}
        elif node[1] == 'output':
            color_map.append(1)
            output_list.append(node[0])
            attr_dict = {'pos': (0.5, 0.1)}

        else:
            rest.append(node[0])
            color_map.append(2)

        G.add_node(node[0])
        



    for gene in genome.genes:
        if gene.enabled:
            G.add_edge(gene.in_node, gene.out, weight=gene.weight)
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos,
                           nodelist = sensor_list,
                           node_color='g')

    nx.draw_networkx_nodes(G, pos,
                           nodelist = output_list,
                           node_color='r')

    nx.draw_networkx_nodes(G, pos,
                           nodelist = rest,
                           node_color='b')


    nx.draw_networkx_edges(G, pos)
    # nx.draw(G, pos, with_labels=True, font_weight="bold", cmap=plt.cm.Blues)
    plt.show()


if __name__ == "__main__":
    genome = Genome()
    genome.initialize(6, 5)
    
    for i in range(20):

        genome.add_node(101)
        genome.add_connection(103)
        genome.add_connection(105)
        genome.add_connection(103)
        genome.add_connection(105)
        genome.add_connection(103)
        genome.add_connection(105)

    plot_genome(genome)
