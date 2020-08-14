from main import *
import random
import numpy as np
import matplotlib.pyplot as plt
import os


colors = ["gray", "yellowgreen", "blue", "brown", "deeppink", "red", "coral", "peru", "darkorange", "gold", "darkkhaki",
          "yellow", "forestgreen", "lime", "darkslategray", "cyan", "dodgerblue", "royalblue", "indigo", "purple"]
figure_size = (20, 10.65625)
figure_legend_loc = 'upper right'
hand_size = DurakEnv.HAND_SIZE
nfsp_saved_models_dir = os.path.join(os.getcwd(), "NFSP-models")
ppo_saved_models_dir = os.path.join(os.getcwd(), "PPOParams")


def plot(x_axis, y_axes, title: str, x_label: str, y_label: str, legend: bool):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for plot_label in y_axes:
        color, values = y_axes[plot_label]
        plt.plot(x_axis, values, color=color, label=plot_label)
    if legend:
        plt.legend()


def graph(dir, title, epochs=100, training_games_per_epoch=50, test_games_per_epoch_vs_test_players=500):
    cumulative_training_games_per_epoch = list(range(
        training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    loss_ratio_vs_1_random = []
    loss_ratio_vs_3_random = []
    trained = TrainedNFSPPlayer(hand_size, 'trained')
    random1 = RandomPlayer(hand_size, 'random-1')
    random2 = RandomPlayer(hand_size, 'random-2')
    random3 = RandomPlayer(hand_size, 'random-3')
    for epoch in range(epochs):
        trained.load_model(dir+'epoc-' + str(epoch + 1))
        loss_ratios = do_test_games(
            [trained, random1], test_games_per_epoch_vs_test_players)
        loss_ratio_vs_1_random.append(loss_ratios[trained.name])
        loss_ratios = do_test_games(
            [trained, random1, random2, random3], test_games_per_epoch_vs_test_players)
        loss_ratio_vs_3_random.append(loss_ratios[trained.name])

    learning_player_info = {"VS. 1 random": (
        colors[1], loss_ratio_vs_1_random), "VS. 3 randoms": (
        colors[2], loss_ratio_vs_3_random)}
    plot(cumulative_training_games_per_epoch, learning_player_info,
         title, "Number of training games", "Loss Ratio", True)
    plt.show()


if __name__ == '__main__':
    graph("train_against_nfsp/", "NFSP trained against another NFSP")
    graph("train_against_prev_iter/", "NFSP trained against previous iterations")
    graph("train_against_one_random/", "NFSP trained against random player")
