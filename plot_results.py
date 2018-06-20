import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_results(data):
    plt.figure()
    plt.plot(np.arange(1, len(data)+1), data)
    plt.ylabel("Score")
    plt.xlabel("Game")

    plt.show()


def plot_smoothed_results(data):
    N = len(data)

    smooth_scale = 0

    data_smoothed = []
    for index, d in enumerate(data):
        data_smoothed.append(np.mean(data[index - smooth_scale:index + smooth_scale])) \
            if index > smooth_scale else data_smoothed.append(d)

    p = np.poly1d(np.polyfit(np.linspace(0, len(data), len(data)), data_smoothed, 100))

    xlabels = np.linspace(0, N, len(data_smoothed))

    # plt.plot(xlabels, data_smoothed, label="smoothed rewards")
    plt.plot(xlabels, p(np.linspace(0, len(data), len(data))), label="polynomial fit")
    plt.plot(xlabels, data, label="Score")
    plt.xlabel("Game")
    plt.ylabel("Fitness")
    plt.legend(loc=0)
    plt.show()


def plot_neat(data, label="fitness"):
    best = []
    median = []
    mean = []
    for generation in data:
        best.append(np.max(generation))
        median.append(np.median(generation))
        mean.append(np.mean(generation))

    plt.figure()
    plt.plot(np.arange(1, len(best) + 1), best, label="max")
    plt.plot(np.arange(1, len(best) + 1), median, label="median")
    plt.plot(np.arange(1, len(best) + 1), mean, label="mean")
    plt.xlabel("Generation")
    plt.ylabel(label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = np.genfromtxt("log.csv", dtype=float, delimiter=',', skip_header=True)
    plot_results(data[:, 1])
    # plot_smoothed_results(data[:, 1])
    # data = pickle.load(open("results/fitnesses.pkl", "rb"))
    # plot_neat(data, label="fitness")

    # data = pickle.load(open("results/scores.pkl", "rb"))
    # plot_neat(data, label="score")
