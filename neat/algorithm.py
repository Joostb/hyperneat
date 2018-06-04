from copy import deepcopy

import numpy as np

from neat.genome import Genome
from visualization.visualization import plot_genome


def crossover(parent_1, parent_2, fitness_1, fitness_2):
    # determine the fittest parent
    if fitness_2 > fitness_1:
        fit_parent = deepcopy(parent_2)
        parent_2 = deepcopy(parent_1)
        parent_1 = fit_parent

    child_genes = cross_genes(parent_1, parent_2)

    # how do we know the nodes?
    child_nodes = generate_nodes(child_genes, parent_1, parent_2)

    child_genome = Genome()
    child_genome.genes = child_genes
    child_genome.nodes = child_nodes
    child_genome.nb_sensor = parent_1.nb_sensor
    child_genome.nb_output = parent_1.nb_output

    return child_genome


def generate_nodes(genes, parent_1, parent_2):
    """
    Loop through all genes, and find out what nodes are used, and their type

    :param genome:
    :return:
    """
    nodes = []
    index_node = []
    for gene in genes:         
        if gene.in_node not in index_node:
            index_node.append(gene.in_node)
        if gene.out not in index_node:
            index_node.append(gene.out)
            
    nb_nodes = len(index_node)
    
    for i in range(nb_nodes):
        if i < parent_1.nb_sensor:
            nodes.append((i, 'sensor'))
        elif i <= parent_1.nb_output+parent_1.nb_sensor:
            nodes.append((i, 'output'))
        else :
             nodes.append((i, 'hidden'))
            
    return nodes 
        


def cross_genes(parent_1: Genome, parent_2: Genome):
    # assume p_1 is the fittest parent
    # this function tries to cross parent 1 and 2 into a new child!

    # for more understanding, please see figure 4, that should explain a lot
    p_1 = 0
    p_2 = 0
    new_gene_list = []
    matching_genes_prob = 0.5
    while True:

        if p_1 >= len(parent_1.genes):
            # we have taken all interesting parts of the fittest genome
            # there are no more overlapping parts
            # so we are done!
            return new_gene_list

        if p_2 >= len(parent_2.genes):

            # we have reached the end of p_2
            # so the rest is excess of p_1, and we want to use it
            for gene in parent_1.genes[p_1:]:
                new_gene_list.append(gene)
            return new_gene_list

        if parent_1.genes[p_1].innov == parent_2.genes[p_2].innov:
            # in this case they are the same and the gene is taken with some probability
            if np.random.rand() > matching_genes_prob:
                new_gene_list.append(deepcopy(parent_1.genes[p_1]))
            p_1 += 1
            p_2 += 1
        elif parent_1.genes[p_1].innov > parent_2.genes[p_2].innov:
            # this is a disjoint part (figure 4, genes 6,7 of parent 2)
            # or it is the excess part,
            #  since p_1 is fitter, we are not interested in this part
            p_2 += 1
        elif parent_1.genes[p_1].innov < parent_2.genes[p_2].innov:
            # here we have a disjoint part of p_1
            # since p_1 is fitter, we want to take these genes!
            new_gene_list.append((parent_1.genes[p_1]))
            p_1 += 1

def delta(parent_1, parent_2):
    """
    Calculate the number of excess and disjoint weights
    and  calculate the mean distance between the weights of matching gens.
    :param parent_1:
    :param parent_2:
    :return:
    """
    p_1 = 0
    p_2 = 0
    disjoint = 0
    excess = 0
    weight_difference = 0
    matching = 0
    while True:

        if p_1 >= len(parent_1.genes):
            excess += len(parent_2.genes) - p_2
            break

        elif p_2 >= len(parent_2.genes):
            # we have taken all interesting parts of the fittest genome
            # there are no more overlapping parts
            # so we are done!
            excess += len(parent_1.genes) - p_1

            break


        elif parent_1.genes[p_1].innov == parent_2.genes[p_2].innov:
            matching += 1
            weight_difference += abs(parent_1.genes[p_1].weight - parent_2.genes[p_2].weight)
            # in this case they are the same and the gene is taken with some probability
            p_1 += 1
            p_2 += 1
        elif parent_1.genes[p_1].innov > parent_2.genes[p_2].innov:
            # this is a disjoint part (figure 4, genes 6,7 of parent 2)
            # or it is the excess part,
            #  since p_1 is fitter, we are not interested in this part
            if len(parent_2.genes) == p_2:
                excess += 1
                p_1 += 1
            else:
                disjoint += 1
                p_2 += 1
        elif parent_1.genes[p_1].innov < parent_2.genes[p_2].innov:
            # here we have a disjoint part of p_1
            # since p_1 is fitter, we want to take these genes!
            if len(parent_1.genes) == p_1:
                excess += 1
                p_2 += 1
            else:
                disjoint += 1
                p_1 += 1
    N = max(len(parent_1.genes), len(parent_2.genes))

    print('matching')
    if matching == 0:

        # todo: matching is sometimes 0, this should not happen
        # todo: figure out where the mistake is
        plot_genome(parent_1)
        plot_genome(parent_2)
        [print(g) for g in (parent_1.genes)]

        print('---------')
        [print(g) for g in (parent_2.genes)]
    print(matching)
    return excess / N, disjoint /N, weight_difference / matching

def distance(parent_1, parent_2, c_1=1.0, c_2=1.0, c_3=0.4):
    E, D, W = delta(parent_1, parent_2)

    return c_1 * E + c_2 * D + c_3 * W




if __name__ == "__main__":
    genome = Genome()
    genome.initialize(3, 2)
    
    genome.add_node(20)
    genome.add_connection(22)


    genome_2 = Genome()
    genome_2.initialize(3, 2)
    genome_2.add_connection(25)
    genome_2.add_connection(28)
    genome_2.add_node(50)

    print(delta(genome, genome_2))

    #plot_genome(genome_2)
    
    # child = crossover(genome, genome_2, 0.5, 0.5)
    #
    # plot_genome(child)
    # print('parent 1')
    # ([print(gene) for gene in genome.genes])
    # ([print(gene) for gene in genome.nodes])
    # print('parent 2')
    # ([print(gene) for gene in genome_2.genes])
    # ([print(gene) for gene in genome_2.nodes])
    # print('child')
    # ([print(gene) for gene in child.genes])
    # ([print(gene) for gene in child.nodes])
    
    '''genome = Genome()
    genome.initialize(6, 5)
    plot_genome(genome)

    genome_2 = Genome()
    genome_2.initialize(6, 5)

    innovation_number = genome_2.innovation_number

    for i in range(3):
        genome.add_connection(innovation_number)
        innovation_number += 1

    genome.add_node(innovation_number)

    cross_genes(genome_2, genome)

    print('network')
    inputs = np.random.normal(size=(5,))
    print(inputs)
    print(genome.evaluate_input(inputs))

    # print(genome.nodes)
    # print(genome.innovation_number)
    # # print((genome.genes))
    ([print(gene) for gene in genome.genes])'''

    # todo complete the crossover step, and check if it works
    # todo crossover
    # todo network evaluation
