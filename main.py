from flappybird.game import FlappyGame, normalize_state
import numpy as np

from neat.genome import Genome


def evolve_flappy():
    gamePool = [FlappyGame(return_rgb=False) for _ in range(population_size)]
    states = [game.get_state() for game in gamePool]

    n_features = len(states[0])

    # This is actually 2 things, doing nothing or flying up. Depending on our implementation we could change it to 1?
    n_actions = len(gamePool[0].valid_actions)

    population = [Genome() for _ in range(population_size)]
    for genome in population:
        genome.initialize(n_features, n_actions)

    for _ in range(n_epochs):
        best_fitness = -float('inf')
        for i, genome in enumerate(population):

            # Play the game to get the fitness
            fitness = 0

            while True:
                # Evaluate the current input and perform the best action
                network_input = normalize_state(states[i])
                action_values = genome.evaluate_input(network_input)

                best_action = np.argmax(action_values)
                best_action = gamePool[i].valid_actions[best_action]

                reward, state, done = gamePool[i].do_action(best_action)
                fitness += reward

                states[i] = state
                if done:
                    states[i] = gamePool[i].reset()
                    genome.reset_activations()
                    print("Fitness:", fitness)
                    if fitness > best_fitness:
                        best_fitness = fitness
                    break

            # Evolve the genome
            # TODO: implement this

        print("Best Epoch Fitness:", best_fitness)


if __name__ == "__main__":
    population_size = 10
    n_epochs = 100
    evolve_flappy()
