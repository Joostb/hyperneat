from flappybird.game import FlappyGame, normalize_state
import numpy as np
import matplotlib.pyplot as plt

from neat_algoritm.genome import Genome
from neat_algoritm.evolving import evolve


def evolve_flappy():
    gamePool = [FlappyGame(return_rgb=False, display_screen=False) for _ in range(population_size)]
    states = [game.get_state() for game in gamePool]

    n_features = len(states[0])

    # This is actually 2 things, doing nothing or flying up. Depending on our implementation we could change it to 1?
    n_actions = len(gamePool[0].valid_actions)

    innovation_number = n_features * n_actions + 1
    representatives = []
    
    global_fitness = [0,0]

    population = [Genome() for _ in range(population_size)]
    for genome in population:
        genome.initialize(n_features, n_actions)

    epoch_values = []
    for _ in range(n_epochs):

        best_fitness = -float('inf')

        # calculate fitness
        print('calculating fitness')
        genome_values = []
        for i, genome in enumerate(population):

            # Play the game to get the fitness
            fitness = 0
            while True:
                # Evaluate the current input and perform the best action
                network_input = normalize_state(states[i])
                action_values = genome.evaluate_input(network_input)
                genome_values.append(action_values)
                    
                #print(action_values)
                best_action = np.argmax(action_values)
                best_action = gamePool[i].valid_actions[best_action]

                reward, state, done = gamePool[i].do_action(best_action)

                fitness += reward 

                states[i] = state
                if done:
                    states[i] = gamePool[i].reset()
                    genome.reset_activations()
                    fitness += 6
                    # print("Fitness:", fitness)
                    if fitness > best_fitness:
                        best_fitness = fitness
                    break

            genome.fitness_number = fitness

        epoch_values.append(np.array(genome_values))

        print('crossing and mutating genes')
        population, representatives, innovation_number, global_fitness = evolve(population, representatives, innovation_number, global_fitness)

        print('generation ', _)
        print("Best Epoch Fitness:", best_fitness)

    for i, values in enumerate(epoch_values):
        plt.plot(values[:, 0], label="e{}_up".format(i))
        plt.plot(values[:, 1], label="e{}_noop".format(i))
    plt.title("Activations over one epoch for one specie")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    population_size = 100
    n_epochs = 100
    evolve_flappy()
