import numpy as np
from neat_algoritm.gene import Gene
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

        self.fitness_number = 0

    def fitness(self):
        # todo
        self.fitness_number = np.random.rand()
        return self.fitness_number

    def initialize(self, n_in, n_out):

        self.nb_sensor = n_in
        self.nb_output = n_out

        self.nodes = []
        for i in range(n_in):
            self.nodes += [(i, 'sensor')]

        for i in range(n_in, n_in + n_out):
            self.nodes += [(i, 'output')]

        for i, node in enumerate(self.nodes):
            if node[1] == 'sensor':
                self.input_neurons.append(i)
                for j, output in enumerate(self.nodes):
                    if output[1] == 'output':
                        weight = np.random.normal()

                        self.genes.append(Gene(i, j, weight, True, self.innovation()))
            elif node[1] is 'output':
                self.output_neurons.append(i)

        self.neuron_activations = np.zeros(shape=(len(self.nodes),))

    def innovation(self):
        self.innovation_number += 1
        return self.innovation_number - 1

    def add_connection(self, innovation_number):
        """
        Add a connection to the genome
        :innovation_number:
        :return:
        """

        out_node = np.random.choice(len(self.nodes))

        nodes = deepcopy(self.nodes)
        nodes = np.random.permutation(nodes)

        for index, in_node in nodes:
            for gene in self.genes:
                if gene.in_node != index and in_node != 'sensor' and self.connection_exist(gene.in_node, out_node) == False:  
                    new_gene = Gene(int(index), out_node, np.random.normal(), True, innovation_number)
                    self.genes.append(new_gene)
                    return new_gene
                
        return None
                
    def connection_exist(self, in_node, out_node):
        """
        Check if a connection between two nodes alredy exist or not
        :in_node:
        :out_node:
        :return:
        """
        for gene in self.genes: 
            if gene.in_node == in_node and gene.out == out_node:
                return True 
            if gene.in_node == out_node and gene.out == in_node:
                return True
        return False

    def add_node(self, innovation_number):
        """
        Add a node to the genome
        :innovation_number:
        :return:
        """
        connection = np.random.choice(self.genes)
        connection.enabled = False

        node = (len(self.nodes), 'hidden')

        self.nodes.append(node)
        self.neuron_activations = np.append(self.neuron_activations, 0)

        new_gene_in = Gene(connection.in_node, node[0], 1, True, innovation_number)
        new_gene_out = Gene(node[0], connection.out, connection.weight, True, innovation_number + 1)

        self.genes.append(new_gene_in)
        self.genes.append(new_gene_out)

        return new_gene_in, new_gene_out

    def mutate_weights(self):
        """
        Mutate the weight according some percentage
        """

        for gene in self.genes:
            # we want the updates to be gaussian noise, not uniform
            # since big changes radically change the 'meanng
            random_update = np.random.normal()
            if np.random.rand() < 0.9:
                gene.weight += random_update * 0.01
            else:
                gene.weight = random_update

    def evaluate_input(self, inputs, steps=2):
        """
        To evaluate the network and return the activation number of the output
        :inputs:
        :step:
        :return:
        """
        
        new_activations = self.neuron_activations  # By reference, no need to set it again
        new_activations[:len(inputs)] += inputs

        new_activations = self.step_network_evaluation(self.input_neurons, new_activations)
        
        output_layer = new_activations[self.output_neurons]

        return output_layer

    def step_network_evaluation(self, input_neurons, activations, activation_function=sigmoid):
        """
        Perfom the evaluation of the network
        :input_neurons:
        :activations:
        :activation_function:
        :return:
        """
        network_new = np.zeros(shape=activations.shape)
        
        for node in self.nodes:
            if node[1] == 'sensor':
                network_new[node[0]] = activations[node[0]]
            else:
                if node[1] == 'hidden':
                    connections = []
                    for gene in self.genes:
                        if node[0] == gene.out:
                            connections.append(gene)
                            
                    for connection in connections:
                        network_new[node[0]] += activations[gene.in_node] * connection.weight * connection.enabled
        
        for node in self.nodes:
            if node[1] == 'output':
                connections = []
                for gene in self.genes:
                    if node[0] == gene.out:
                        connections.append(gene)
                        
                for connection in connections:
                    network_new[node[0]] += activations[gene.in_node] * connection.weight * connection.enabled
        
        
        return activation_function(network_new)

    def reset_activations(self):
        self.neuron_activations = np.zeros(shape=(len(self.nodes),))
