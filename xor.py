from flappybird.game import FlappyGame, normalize_state
import numpy as np
import matplotlib.pyplot as plt

from neat.genome import Genome
from neat.evolving import evolve


def evolve_xor():
    n_features = 2

    n_actions = 2  # Either 0 or 1, such that we don't have to do thresholding

    innovation_number = n_features * n_actions + 1
    representatives = []

    xor_input = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    xor_output = [
        0,
        1,
        1,
        0
    ]

    population = [Genome() for _ in range(population_size)]
    for genome in population:
        genome.initialize(n_features, n_actions)

    epoch_fitnesses = []
    epoch_activations = []
    for _ in range(n_epochs):

        best_fitness = 0

        # calculate fitness
        print('calculating fitness')

        # buffer for storing fitness per genome
        genome_fitnesses = []
        genome_activations = []


        for i, genome in enumerate(population):

            # Run XOR inputs
            fitness = 1

            # rebuild the number of nodes etc
            genome.reset_activations()
            genome_activation = []

            for X, y in zip(xor_input, xor_output):
                # Evaluate the current input and perform the best action
                output = genome.evaluate_input(X, steps=2)
                y_pred = np.argmax(output)
                genome_activation.append(output[y_pred])

                if y_pred == y:
                    fitness += 1

                genome.reset_activations()

            genome_fitnesses.append(fitness) # fitness
            genome_activations.append(np.array(genome_activation))
            if fitness > best_fitness:
                best_fitness = fitness

            genome.fitness_number = fitness

        epoch_fitnesses.append(np.array(genome_fitnesses))
        epoch_activations.append(np.array(genome_activations))

        print('crossing and mutating genes')
        population, representatives, innovation_number = evolve(population, representatives, innovation_number, disable_crossover=True)

        print('generation ', _)
        print("Best Epoch Fitness:", best_fitness)

    epoch_fitnesses = np.array(epoch_fitnesses)
    print(epoch_fitnesses)
    for i in range(epoch_fitnesses.shape[1]):
        plt.plot(epoch_fitnesses[:, i] - 1, label="specie {}".format(i))

    plt.title("Fitness per epoch per specie")
    plt.xlabel("epoch")
    plt.ylabel("fitness")
    plt.legend()
    plt.show()

    epoch_activations = np.array(epoch_activations)
    for specie in range(epoch_activations.shape[1]):
        for datapoint in range(epoch_activations.shape[2]):
            plt.plot(epoch_activations[:, specie, datapoint], label="specie{}_{}".format(specie, xor_input[datapoint]))
    plt.title("Activation per epoch per specie per XOR input")
    plt.xlabel("epoch")
    plt.ylabel("activation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    population_size = 5
    n_epochs = 400
    evolve_xor()
