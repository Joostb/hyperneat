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

        out_node = np.random.choice(len(self.nodes))

        nodes = deepcopy(self.nodes)
        nodes = np.random.permutation(nodes)

        for index, in_node in nodes:

            for gene in self.genes:
                if gene.in_node != index and in_node[1] != 'sensor':  # can there be loops 4 -> 4
                    new_gene = Gene(int(index), out_node, np.random.normal(), True, innovation_number)
                    self.genes.append(new_gene)
                    return new_gene

    def add_node(self, innovation_number):
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


        for gene in self.genes:
            # we want the updates to be gaussian noise, not uniform
            # since big changes radically change the 'meanng
            random_update = np.random.normal()
            if np.random.rand() < 0.9:
                gene.weight += random_update * 0.01
            else:
                gene.weight = random_update

    def evaluate_input(self, inputs, steps=2):


        new_activations = self.neuron_activations  # By reference, no need to set it again
        # new_activations[:len(inputs)] = inputs
        new_activations[:len(inputs)] += inputs

        for i in range(steps):
            new_activations = self.step_network_evaluation(self.input_neurons, new_activations)
            # new_activations = self.step_network_evaluation(self.hidden_neurons, new_activations)

        output_layer = new_activations[self.output_neurons]

        return output_layer

    def step_network_evaluation(self, input_neurons, activations, activation_function=sigmoid):

        # todo: debug this function
        # here we want to calculate the new activations for all neurons, based on the old activations.
        
#        print(input_neurons)
#        print(activations)
        network_new = np.zeros(shape=activations.shape)
#        print(network_new.shape)
#        [print(g) for g in self.genes]
#        print(self.nodes)
        
        for gene in self.genes:
            #print(gene.out)
            network_new[gene.out] += activations[gene.in_node] * gene.weight * gene.enabled

        # numpy can apply the function elementwise, without looping!
        return activation_function(network_new)

        ## WRONG ##
        ## genes denote connections **between** neurons, not neurons themselves

        for neuron in input_neurons:
            gene = self.genes[neuron]
            activations[gene.out] += activations[gene.in_node] * gene.weight * gene.enabled

        for out_neuron in set([self.genes[neuron].out for neuron in input_neurons]):
            activations[out_neuron] = activation_function(activations[out_neuron])

        return activations

    def reset_activations(self):
        self.neuron_activations = np.zeros(shape=(len(self.nodes),))
