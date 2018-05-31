import networkx as nx
import matplotlib.pyplot as plt

from neat import Genome


def plot_genome(genome: Genome):
    G = nx.DiGraph()

    for node_nr, _ in enumerate(genome.nodes):
        G.add_node(node_nr)

    for gene in genome.genes:
        if gene.enabled:
            G.add_edge(gene.in_node, gene.out, weight=gene.weight)

    nx.draw(G, with_labels=True, font_weight="bold")
    plt.show()


if __name__ == "__main__":
    genome = Genome()
    genome.initialize(6, 5)

    plot_genome(genome)
