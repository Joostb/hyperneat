import numpy as np
from neat.gene import Gene
from copy import deepcopy


class Genome:
    def __init__(self):
        # eg : self.nodes = 'sensor' 'sensor' 'output'
        self.nodes = []

        # list of Genes
        self.genes = []

        self.innovation_number = 0

    def initialize(self, n_in, n_out):
        
        self.nb_sensor = n_in
        self.nb_output = n_out
        
        self.nodes = []        
        for i in range(n_in):
            self.nodes += [(i, 'sensor')]
            
        for i in range(n_in, n_in+n_out):
            self.nodes += [(i, 'output')] 

        for i, sensor in enumerate(self.nodes):
            if sensor[1] == 'sensor':
                for j, output in enumerate(self.nodes):
                    if output[1] == 'output':
                        weight = np.random.rand()

                        self.genes.append(Gene(i, j, weight, True, self.innovation()))

    def innovation(self):
        self.innovation_number += 1
        return self.innovation_number - 1

    def add_connection(self, innovation_number):

        out_node = np.random.choice(len(self.nodes))

        nodes = deepcopy(self.nodes)
        nodes = np.random.permutation(nodes)

        for index, in_node in nodes:

            for gene in self.genes:
                if gene.in_node != index and in_node[1] != 'sensor':  # can there be loops 4 -> 4
                    new_gene = Gene(int(index), out_node, np.random.rand(), True, innovation_number)
                    self.genes.append(new_gene)
                    return new_gene

    def add_node(self, innovation_number):
        connection = np.random.choice(self.genes)
        connection.enabled = False

        node = (len(self.nodes), 'hidden')

        self.nodes.append(node)
        
        new_gene_in = Gene(connection.in_node, node[0], 1, True, innovation_number)
        new_gene_out = Gene(node[0], connection.out, connection.weight, True, innovation_number + 1)

        self.genes.append(new_gene_in)
        self.genes.append(new_gene_out)

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
