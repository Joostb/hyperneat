import neat

import pickle

import numpy as np
from tqdm import tqdm

import visualize
from flappybird.game_custom import game

SCORES = []
FITNESSES = []


def eval_genomes(genomes, config):
    global SCORES, FITNESSES
    fitnesses = np.zeros(len(genomes))
    scores = np.zeros(len(genomes))
    for i, (genome_id, genome) in enumerate(tqdm(genomes, desc="Evaluating Genomes", unit="genomes", leave=False)):
        genome.fitness, game_score = game(genome, config, display_screen=False)
        fitnesses[i] = genome.fitness
        scores[i] = game_score
    print("High-score:", np.max(scores))
    SCORES.append(scores)
    FITNESSES.append(fitnesses)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 50)

    print("\n Best Genome: \n{}".format(winner))
    fitness, score = game(winner, config, display_screen=True)
    print("\tFitness: {} \n\tScore: {}".format(fitness, score))

    visualize.draw_net(config, winner, view=True,
                       node_names={-1: 'y_flappy', -2: 'dist_pipe', -3: "y_top_pipe", -4: "y_bottom_pipe", 0: 'flap'})
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    pickle.dump(winner, open("best_genome.pkl", "wb"))


if __name__ == "__main__":
    run("config")
    pickle.dump(SCORES, open("scores.pkl", "wb"))
    pickle.dump(FITNESSES, open("fitnesses.pkl", "wb"))
