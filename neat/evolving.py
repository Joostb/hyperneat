

import numpy as np

from neat.genome import Genome
from visualization.visualization import plot_genome
from neat.algorithm import distance, crossover

def evolve():
    input = 10
    output = 5
    genomes = [Genome() for _ in range(100)]
    [g.initialize(input, output) for g in genomes]
    generations = 20
    representatives = []

    delta_t = 3.0
    innovation_number = 100

    for k in range(generations):
        species = [[] for _ in representatives]
        children = []
        for genome in genomes:
            belongs = False
            for i, rep in enumerate(representatives):
                if distance(rep, genome) < delta_t:
                    species[i].append(genome)
                    belongs = True
                    break
            if not belongs:
                representatives.append(genome)
                species.append([genome])

        genome_fitness = [fitness(g) for g in genomes]
               


        for specy in species:
            for genome in specy:
                if np.random.rand() < 0.8:
                    genome.mutate_weights()

                if np.random.rand() < 0.03:
                    genome.add_node(innovation_number)
                    innovation_number += 2

                if np.random.rand() < 0.05:
                    genome.add_connection(innovation_number)
                    innovation_number += 1



        for specy in species:
            for genome in specy:
                if np.random.rand() < 0.25:
                    if np.random.rand() < 0.001:
                        # todo change the fitness...
                        children.append(crossover(genome, np.random.choice(genomes),1,2))
                    else:
                        children.append(crossover(genome, np.random.choice(specy),1,2))

        


        genomes += children
        print('genomes')
        print(len(genomes))



def fitness(genome):
    pass 

def fitness_sharing(specy, genome, threshold):
    
    sharing_function = len(specy)     
    
    return fitness(genome) / sharing_function


if __name__ == '__main__':
    evolve()