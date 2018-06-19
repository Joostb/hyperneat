import matplotlib.pyplot as plt
import numpy as np


def plot_results(data):
    plt.figure()
    plt.plot(np.arange(1, len(data)+1), data)
    plt.ylabel("Fitness")
    plt.xlabel("Game")

    plt.show()


def plot_smoothed_results(data):
    N = len(data)

    smooth_scale = 3

    data_smoothed = []
    for index, d in enumerate(data):
        data_smoothed.append(np.mean(data[index - smooth_scale:index + smooth_scale])) \
            if index > smooth_scale else data_smoothed.append(d)

    p = np.poly1d(np.polyfit(np.linspace(0, len(data), len(data)), data_smoothed, 100))

    xlabels = np.linspace(0, N, len(data_smoothed))

    # plt.plot(xlabels, data_smoothed, label="smoothed rewards")
    plt.plot(xlabels, p(np.linspace(0, len(data), len(data))), label="smoothed reward")
    plt.plot(xlabels, data, label="Actual")
    plt.xlabel("Game")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    data = np.genfromtxt("log.csv", dtype=float, delimiter=',', skip_header=True)
    # plot_results(data[:, 1])
    plot_smoothed_results(data[:, 1])
