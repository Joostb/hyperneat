import numpy as np

from neat.genome import Genome
from visualization.visualization import plot_genome
from neat.algorithm import distance, crossover
from tqdm import tqdm


def evolve(genomes, representatives, innovation_number, delta_t=3.0):
    # input = 10
    # output = 5
    # genomes = [Genome() for _ in range(100)]
    # [g.initialize(input, output) for g in genomes]
    # generations = 50000
    # representatives = []
    #
    # delta_t = 2.0
    # innovation_number = 100

    # in parallel or something

    species = [[] for _ in representatives]

    children = []
    total_fitness = 0

    # calculate the different species
    for genome in genomes:
        belongs = False
        for i, rep in enumerate(representatives):
            if distance(rep, genome) < delta_t:
                species[i].append(genome)
                total_fitness += genome.fitness_number
                belongs = True
                # if k > 1000:
                #     print(distance(rep, genome))
                break
        if not belongs:
            representatives.append(genome)
            species.append([genome])
            total_fitness += genome.fitness_number

    strong_species = []

    # crossover
    for specy in species:
        specy_fitness = sum([g.fitness_number for g in specy]) / total_fitness
        allowed_offspring = int(len(genomes) * specy_fitness)

        strong = eliminate_weakest(specy)
        strong_species.append(strong)

        # for genome in specy:
        for i in range(allowed_offspring - len(strong)):
            parent_1 = np.random.choice(strong)

            if np.random.rand() < 0.001:
                s = np.random.randint(len(species))
                parent_2 = np.random.choice(species[s])

                # todo change the fitness...
                children.append(crossover(parent_1, parent_2, parent_1.fitness_number, parent_2.fitness_number))
            else:

                parent_2 = np.random.choice(strong)
                children.append(crossover(parent_1, parent_2, parent_1.fitness_number, parent_2.fitness_number))

    # genome_fitness = [fitness(g) for g in genomes]

    # mutation
    for specy in strong_species:
        for genome in specy:
            if np.random.rand() < 0.8:
                genome.mutate_weights()

            if np.random.rand() < 0.03:
                genome.add_node(innovation_number)
                innovation_number += 2

            if np.random.rand() < 0.05:
                genome.add_connection(innovation_number)
                innovation_number += 1

    new_genomes = children
    for specy in strong_species:
        new_genomes += specy
    # create new generation
    genomes = new_genomes

    print('--------------')

    print('genomes')
    print(len(genomes))

    print('species')
    print(len(species))

    return genomes, representatives, innovation_number

    if len(species) > 1:
        print('>1 species')

    print('--------------')

    print('genomes')
    print(len(genomes))
    print(len(genomes[10].genes))
    print(len(genomes[40].genes))
    print(len(genomes[90].genes))
    print('species')
    print(len(species))


def eliminate_weakest(specy, percentage=0.25):
    sort = sorted(specy, key=lambda g: g.fitness_number)

    # eliminate weakest 25%
    return sort[int(len(sort) * percentage):]


# def fitness(genome):
#     pass
#
#
# def fitness_sharing(specy, genome, threshold):
#     sharing_function = len(specy)
#
#     return fitness(genome) / sharing_function


if __name__ == '__main__':
    evolve()
