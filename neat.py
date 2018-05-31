

import numpy as np

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

        nodes = self.nodes.copy()
        nodes = np.random.permutation(list(enumerate(nodes)))


        for index, in_node in nodes:

            for gene in self.genes:
                if gene.in_node != index and in_node != 'sensor': # can there be loops 4 -> 4
                    new_gene = Gene(int(index), out_node, np.random.rand(), True, self.innovation_number)
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
        network_old = [0 for i in range(len(self.nodes))]
        for i, value in enumerate(inputs):
            network_old[i] = value


        network_new = self.step_network_evaluation(network_old)

        return network_new



    def step_network_evaluation(self, network_old):
        network_new = [0] * len(self.nodes)


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




genome = Genome()
genome.initialize(5,2)

for i in range(3):
    genome.add_node(1)
    genome.add_connection(1)


print('network')
print(genome.evaluate_input([1,2,3,4,5]))

print(genome.nodes)
print(genome.innovation_number)
# print((genome.genes))
([print(gene) for gene in genome.genes])


# todo crossover
# todo network evaluation
# todo nice graph printing function