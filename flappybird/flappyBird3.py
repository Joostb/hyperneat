import os
import pickle

import neat

from flappybird.game_custom import game

SCORE = 0

GENERATION = 0
MAX_FITNESS = 0
BEST_GENOME = 0


def eval_genomes(genomes, config):
    i = 0
    global SCORE
    global GENERATION, MAX_FITNESS, BEST_GENOME

    GENERATION += 1
    for genome_id, genome in genomes:

        genome.fitness = game(genome, config)
        print("Gen : %d Genome # : %d  Fitness : %f Max Fitness : %f" % (GENERATION, i, genome.fitness, MAX_FITNESS))
        if genome.fitness >= MAX_FITNESS:
            MAX_FITNESS = genome.fitness
            BEST_GENOME = genome
        SCORE = 0
        i += 1


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

winner = pop.run(eval_genomes, 30)

print(winner)

outputDir = './'
os.chdir(outputDir)
serialNo = len(os.listdir(outputDir)) + 1
outputFile = open(str(serialNo) + '_' + str(int(MAX_FITNESS)) + '.p', 'wb')

pickle.dump(winner, outputFile)
