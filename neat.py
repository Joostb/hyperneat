

import numpy as np
from copy import deepcopy

class Genome:
    def __init__(self):
        # eg : self.nodes = 'sensor' 'sensor' 'output'
        self.nodes = []

        # list of Genes
        self.genes = []

        self.innovation_number = 0

    def initialize(self, n_in, n_out):
        self.nodes = ['sensor'] * n_in
        self.nodes += ['output'] * n_out

        for i, sensor in enumerate(self.nodes):
            if sensor == 'sensor':
                for j, output in enumerate(self.nodes):
                    if output == 'output':
                        weight =  np.random.rand()

                        self.genes.append(Gene(i, j, weight, True, self.innovation()))


    def innovation(self):
        self.innovation_number += 1
        return self.innovation_number - 1

    def add_connection(self, innovation_number):

        out_node = np.random.choice(len(self.nodes))

        nodes = deepcopy(self.nodes)
        nodes = np.random.permutation(list(enumerate(nodes)))


        for index, in_node in nodes:

            for gene in self.genes:
                if gene.in_node != index and in_node != 'sensor': # can there be loops 4 -> 4
                    new_gene = Gene(int(index), out_node, np.random.rand(), True, innovation_number)
                    self.genes.append(new_gene)
                    return new_gene

    def add_node(self, innovation_number):
        connection = np.random.choice(self.genes)
        connection.enabled = False

        self.nodes.append('hidden')

        node = len(self.nodes) - 1

        new_gene_in = Gene(connection.in_node, node, 1, True, innovation_number)
        new_gene_out = Gene(node, connection.out, connection.weight, True, innovation_number + 1)

        self.genes.append(new_gene_in)
        self.genes.append((new_gene_out))

        return new_gene_in, new_gene_out



    def evaluate_input(self, inputs):

        network_old = np.zeros(len(self.nodes))
        for i, value in enumerate(inputs):
            network_old[i] = value


        for i in range(10):
            # calculate the input values for all network nodes
            network_new = self.step_network_evaluation(network_old)

            # calculate the activations for all network nodes
            network_new = 1 / (1 + np.exp(-4.9 * network_new))

            network_old = deepcopy(network_new)


        return network_new



    def step_network_evaluation(self, network_old):
        network_new = np.zeros(len(network_old))



        for gene in self.genes:
            network_new[gene.out] += network_old[gene.in_node] * gene.weight * gene.enabled


        return network_new

class Gene:

    def __init__(self, in_node, out, weight,  enabled:bool, innov):
        self.in_node = in_node
        self.out = out
        self.weight = weight
        self.enabled = enabled
        self.innov = innov

    def __str__(self):
        return str([self.in_node, self.out, self.weight, self.enabled, self.innov])


def crossover(parent_1, parent_2, fitness_1, fitness_2):

    # determine the fittest parent
    if fitness_2 > fitness_1:
        fit_parent = deepcopy(parent_2)
        parent_2 = deepcopy(parent_1)
        parent_1 = fit_parent

    child_genes = cross_genes(parent_1, parent_2)

    # how do we know the nodes?
    child_nodes  = generate_nodes(child_genes)

    child_genome = Genome()
    child_genome.genes = child_genes
    child_genome.nodes = child_nodes

    return child_genome



def generate_nodes(genes):
    """
    Loop through all genes, and find out what nodes are used, and their type

    :param genome:
    :return:
    """
    pass

def cross_genes(parent_1:Genome, parent_2:Genome):
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


genome = Genome()
genome.initialize(6,5)

genome_2 = Genome()
genome_2.initialize(6,5)

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
([print(gene) for gene in genome.genes])

# todo complete the crossover step, and check if it works
# todo crossover
# todo network evaluation
# todo nice graph printing function