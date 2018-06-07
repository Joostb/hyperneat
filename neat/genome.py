import numpy as np
from neat.gene import Gene
from copy import deepcopy

sigmoid = lambda x: 1 / (1 + np.exp(-4.9 * x))
relu = lambda x: np.max([0, x])


class Genome:
    def __init__(self):
        # eg : self.nodes = 'sensor' 'hidden' 'output'
        self.nodes = []

        # list of Genes
        self.genes = []

        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []

        self.innovation_number = 0
        self.neuron_activations = None
        self.neuron_weights = None

    def initialize(self, n_in, n_out):
        
        self.nb_sensor = n_in
        self.nb_output = n_out
        
        self.nodes = []        
        for i in range(n_in):
            self.nodes += [(i, 'sensor')]
            
        for i in range(n_in, n_in+n_out):
            self.nodes += [(i, 'output')] 

        for i, node in enumerate(self.nodes):
            if node[1] == 'sensor':
                self.input_neurons.append(i)
                for j, output in enumerate(self.nodes):
                    if output[1] == 'output':
                        weight = np.random.rand()

                        self.genes.append(Gene(i, j, weight, True, self.innovation()))
            elif node[1] is 'output':
                self.output_neurons.append(i)

        self.neuron_activations = np.zeros(shape=(len(self.nodes), ))

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

    def mutate_weights(self):
        for gene in self.genes:
            if np.random.rand() <0.9:
                gene.weight += np.random.normal(scale=0.01)
            else:
                gene.weight = np.random.normal()

    def evaluate_input(self, inputs):
        new_activations = self.neuron_activations  # By reference, no need to set it again
        new_activations[:len(inputs)] = inputs

        new_activations = self.step_network_evaluation(self.input_neurons, new_activations)
        new_activations = self.step_network_evaluation(self.hidden_neurons, new_activations)

        output_layer = new_activations[self.output_neurons]

        return output_layer

    def step_network_evaluation(self, neurons, activations, activation_function=sigmoid):
        for neuron in neurons:
            gene = self.genes[neuron]
            activations[gene.out] += activations[gene.in_node] * gene.weight * gene.enabled

        for out_neuron in set([self.genes[neuron].out for neuron in neurons]):
            activations[out_neuron] = activation_function(activations[out_neuron])

        return activations

    def reset_activations(self):
        self.neuron_activations = np.zeros(shape=(len(self.nodes), ))
