from collections import defaultdict

import numpy as np
from PPONetwork import PPONetwork, PPOModel
import tensorflow as tf
import joblib
from Deck import Deck
from DurakEnv import DurakEnv
from PPOPlayer import PPOPlayer, from_path
from RandomPlayer import RandomPlayer
from DefensivePlayer import DefensivePlayer
from AggressivePlayer import AggressivePlayer
from HumanPlayer import HumanPlayer
import logging
import time
from PPOTrainer import PPOTrainer
import os
import matplotlib.pyplot as plt

NUM_GAMES = 300


def get_sorted_filenames_and_indices():
    files = [(int(f[5:]), f) for f in os.listdir(os.curdir + '/PPOParams/')
             if f.find('model') != -1 and int(f[5:]) % 5000 == 0]
    files.sort(key=lambda x: x[0])
    return list(zip(*files))


def run_against_one_random(params_filename):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "PPO_" + params_filename, 'PPOParams/' + params_filename)
        player2 = RandomPlayer(DurakEnv.HAND_SIZE, "random 1")

        game = DurakEnv([player1, player2], False)
        lost = 0
        for game_index in range(NUM_GAMES):
            state = game.reset()
            game.render()
            steps = 0
            while True:
                turn_player = game.get_turn_player()
                to_attack = game.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done, info = game.step(act)
                state = new_state
                game.render()
                if done:
                    if game.get_loser():
                        if game.get_loser().name == player1.name:
                            lost += 1
                    break

                steps += 1
                if steps > 500:
                    break

        return lost / NUM_GAMES


def run_against_random(params_filename):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "PPO_" + params_filename, 'PPOParams/' + params_filename)
        player2 = RandomPlayer(DurakEnv.HAND_SIZE, "random 1")
        player3 = RandomPlayer(DurakEnv.HAND_SIZE, "random 2")
        player4 = RandomPlayer(DurakEnv.HAND_SIZE, "random 3")

        game = DurakEnv([player1, player2, player3, player4], False)
        lost = 0
        for game_index in range(NUM_GAMES):
            state = game.reset()
            game.render()
            steps = 0
            while True:
                turn_player = game.get_turn_player()
                to_attack = game.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done, info = game.step(act)
                state = new_state
                game.render()
                if done:
                    if game.get_loser():
                        if game.get_loser().name == player1.name:
                            lost += 1
                    break

                steps += 1
                if steps > 500:
                    break

        return lost / NUM_GAMES


def run_against_defensive(params_filename):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "PPO_" + params_filename, 'PPOParams/' + params_filename)
        player2 = DefensivePlayer(DurakEnv.HAND_SIZE, "def 1")
        player3 = DefensivePlayer(DurakEnv.HAND_SIZE, "def 2")
        player4 = DefensivePlayer(DurakEnv.HAND_SIZE, "def 3")

        game = DurakEnv([player1, player2, player3, player4], False)
        lost = 0
        for game_index in range(NUM_GAMES):
            state = game.reset()
            game.render()
            steps = 0
            while True:
                turn_player = game.get_turn_player()
                to_attack = game.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done, info = game.step(act)
                state = new_state
                game.render()
                if done:
                    if game.get_loser():
                        if game.get_loser().name == player1.name:
                            lost += 1
                    break

                steps += 1
                if steps > 500:
                    break

        return lost / NUM_GAMES


def run_against_aggressive(params_filename):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "PPO_" + params_filename, 'PPOParams/' + params_filename)
        player2 = AggressivePlayer(DurakEnv.HAND_SIZE, "agg 1")
        player3 = AggressivePlayer(DurakEnv.HAND_SIZE, "agg 2")
        player4 = AggressivePlayer(DurakEnv.HAND_SIZE, "agg 3")

        game = DurakEnv([player1, player2, player3, player4], False)
        lost = 0
        for game_index in range(NUM_GAMES):
            state = game.reset()
            game.render()
            steps = 0
            while True:
                turn_player = game.get_turn_player()
                to_attack = game.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done, info = game.step(act)
                state = new_state
                game.render()
                if done:
                    if game.get_loser():
                        if game.get_loser().name == player1.name:
                            lost += 1
                    break

                steps += 1
                if steps > 500:
                    break

        return lost / NUM_GAMES


def plot_graph(title, indices, losses, graph_num):
    plt.subplot(graph_num)
    plt.title(title)
    plt.xlabel("Player Model Number")
    plt.ylabel("Loss Ratio")
    plt.ylim(0, 1)
    plt.plot(indices, losses)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(filename='logs/PPOTester_log', level=logging.INFO)
    indices, files = get_sorted_filenames_and_indices()

    # run against randoms, save results, and plot graph
    random_losses = []
    for f in files:
        random_losses.append(run_against_one_random(f))

    plot_graph("PPO models loss ratio against 1 Random Agent", indices, random_losses, graph_num=111)

    # # run against randoms, save results, and plot graph
    # random_losses = []
    # for f in files:
    #     random_losses.append(run_against_random(f))
    #
    # plot_graph("PPO models loss ratio against 3 Random Agents", indices, random_losses, graph_num=111)

    # # run against defensives, save results, and plot graph
    # def_losses = []
    # for f in files:
    #     def_losses.append(run_against_defensive(f))
    #
    # plot_graph("PPO models loss ratio against 3 Defensive Agents", indices, def_losses, graph_num=211)

    # # run against defensives, save results, and plot graph
    # agg_losses = []
    # for f in files:
    #     agg_losses.append(run_against_aggressive(f))
    #
    # plot_graph("PPO models loss ratio against 3 Aggressive Agents", indices, agg_losses, graph_num=111)

    logging.shutdown()
