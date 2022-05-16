"""Plotter useful in games"""
import matplotlib.pyplot as plt

plt.ion()


def gamescore_plotter(scores, mean_scores):
    """Plots the scores and mean scores"""
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
