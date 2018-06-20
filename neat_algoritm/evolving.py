import numpy as np

from neat_algoritm.algorithm import distance, crossover


def evolve(genomes, representatives, innovation_number, delta_t=3.0):
    """
    Main algorithm for evoling a population
    :param genomes:
    :param representatives:
    :param innovation_number:
    :param delta_t:
    :return:
    """
    species = [[] for _ in representatives]

    children = []
    total_fitness = 0
    pop_size = len(genomes)

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

    # remove empty species if they exist
    species = list(filter(lambda sp: sp != [], species))
    
    #check if stagnation 
#    if total_fitness > global_fitness[0] :
#        global_fitness[0] = total_fitness
#    else:
#        if global_fitness[1] > 15:
#            species = stagnation(species, total_fitness)
#            global_fitness[1] = 0
#        else:
#            global_fitness[1] += 1

    strong_species = []
        
    allowed_offsprings = new_species_size(species, pop_size, total_fitness)

    # crossover
    for idx, specy in enumerate(species):
        allowed_offspring = allowed_offsprings[idx]
        strong = eliminate_weakest(specy)

        # cull more weak members
        if len(strong) > allowed_offspring:
            strong = strong[:allowed_offspring]
        strong_species.append(strong)


        # for genome in specy:
        for i in range(allowed_offspring - len(strong)):
            parent_1 = np.random.choice(strong)

            if np.random.rand() < 0.001:
                s = np.random.randint(len(species))
                parent_2 = np.random.choice(species[s])

                children.append(crossover(parent_1, parent_2, parent_1.fitness_number, parent_2.fitness_number))
            else:

                parent_2 = np.random.choice(strong)
                children.append(crossover(parent_1, parent_2, parent_1.fitness_number, parent_2.fitness_number))

    # mutation
    for specy in strong_species:
        for genome in specy:
            if np.random.rand() < 0.8:
                genome.mutate_weights()
            #
            if np.random.rand() < 0.2:
                genome.add_node(innovation_number)
                innovation_number += 2

            if np.random.rand() < 0.2:
                genome.add_connection(innovation_number)
                innovation_number += 1

    # getting new representatives
    representatives = [s[0] for s in filter(lambda s: s != [], strong_species)]

    new_genomes = children
    for specy in strong_species:
        new_genomes += specy
    
        

    # create new generation
    genomes = new_genomes
    #print informations each generations 
    print('--------------')
    print('genomes')
    print(len(genomes))
    print('average size', sum([len(g.genes) for g in genomes]) / len(genomes))
    print('species')
    print(len(species))

    return genomes, representatives, innovation_number


def new_species_size(species, total, total_fitness):
    """
    Calculate the number of genome per species going to the next generation without mutations
    :species:
    :total:
    :total_fitness:
    :return:
    """
    allowed_offsprings = []

    for idx, specy in enumerate(species):
        specy_fitness = sum([g.fitness_number for g in specy]) / total_fitness
        allowed_offspring_old = int(total * specy_fitness)
        allowed_offsprings.append(allowed_offspring_old)

    while sum(allowed_offsprings) != total:
        if sum(allowed_offsprings) < total:
            allowed_offsprings[0] += 1
        else:
            allowed_offsprings[0] -= 1

    if sum(allowed_offsprings) != total:
        print('error')

    return allowed_offsprings


def eliminate_weakest(specy, percentage=0.25):
    """
    Eliminate a certain percentage of genome per species
    :specy:
    :percentage:
    :return:
    """
    percentage = 0.5
    sort = sorted(specy, key=lambda g: g.fitness_number)

    # eliminate weakest 25%
    return sort[int(len(sort) * percentage):]

def fitness_specy(specy, total_fitness):
    """
    Calculate the fitness of genomes per species according the global fitness 
    :specy:
    :total_fitness:
    :return:
    """
    return sum([g.fitness_number for g in specy]) / total_fitness
    
def stagnation(species, total_fitness):
    """
    Select the 2 best species (allowed to reproduce) because of stagnation
    :species:
    :total_fitness:
    :return:
    """
    reproductible_species = []
    
    for idx, specy in enumerate(species):
        if idx == 0 or idx == 1:
            reproductible_species.append(specy)
        else: 
            s_fitness = fitness_specy(specy, total_fitness) 
            if s_fitness > fitness_specy(reproductible_species[0], total_fitness):
                del reproductible_species[0]
                reproductible_species.append(specy)
            else: 
                if s_fitness > fitness_specy(reproductible_species[1], total_fitness):
                    del reproductible_species[1]
                    reproductible_species.append(specy)
                    
    return reproductible_species

if __name__ == '__main__':
    evolve()
