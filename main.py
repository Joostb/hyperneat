from flappybird.game import FlappyGame
import numpy as np

from neat.genome import Genome


def evolve_flappy():
    gamePool = [FlappyGame(return_rgb=False) for _ in range(population_size)]
    states = [game.get_state() for game in gamePool]

    sample_state = states[0]
    n_features = len(sample_state.keys())

    # This is actually 2 things, doing nothing or flying up. Depending on our implementation we could change it to 1?
    n_actions = len(gamePool[0].valid_actions)

    population = [Genome() for _ in range(population_size)]
    for genome in population:
        genome.initialize(n_features, n_actions)

    for _ in range(n_frames):
        for i, genome in enumerate(population):
            # Evaluate the current input and perform the best action
            network_input = list(states[i].values())
            action_values = genome.evaluate_input(network_input)
            print("Action Values:", action_values)
            best_action = np.argmax(action_values)
            best_action = gamePool[i].valid_actions[best_action]
            fitness, state, done = gamePool[i].do_action(best_action)
            states[i] = state
            print("Fitness:", fitness)
            print("Game State:", state)
            if done:
                print("died")
                gamePool[i].reset()

            # Evolve the genome
            # TODO: implement this


if __name__ == "__main__":
    population_size = 1
    n_frames = 1000
    evolve_flappy()