from NFSPTrainedPlayer import TrainedNFSPPlayer
from AggressivePlayer import AggressivePlayer
from DefensivePlayer import DefensivePlayer
from RandomPlayer import RandomPlayer
from HumanPlayer import HumanPlayer
from DurakPlayer import DurakPlayer
from NFSPPlayer import NFSPPlayer
from PPOPlayer import PPOPlayer, from_path

from Types import List
from NFSPTrainer import NFSPTrainer
from PPOTrainer import PPOTrainer
from PPONetwork import PPONetwork
from DurakEnv import DurakEnv

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import random
import shutil
import joblib
import sys
import os


colors = ["gray", "yellowgreen", "blue", "brown", "deeppink", "red", "coral", "peru", "darkorange", "gold", "darkkhaki",
          "yellow", "forestgreen", "lime", "darkslategray", "cyan", "dodgerblue", "royalblue", "indigo", "purple"]
figure_size = (20, 10.65625)
figure_legend_loc = 'upper right'
hand_size = DurakEnv.HAND_SIZE
nfsp_saved_models_dir = os.path.join(os.getcwd(), "NFSP-models")
ppo_saved_models_dir = os.path.join(os.getcwd(), "PPOParams")


def do_test_games(players: List[DurakPlayer], num_games: int, steps_lim=500):
    loss_ratios = {player.name: 0 for player in players}
    env = DurakEnv(players, False)
    print_every = min(num_games // 10, 50)
    for i in range(1, num_games + 1):
        if num_games >= 100:
            if i == 1 or (i % print_every) == 0:
                print("playing test game", i, "out of", num_games)
        state = env.reset()
        done = False
        step = 1
        while not done and (step < steps_lim):
            act = (env.get_turn_player()).get_action(state, env.to_attack())
            state, _, done = env.step(act)
            step += 1
        loser = env.get_loser()
        if loser is not None:
            loss_ratios[loser.name] += 1
    if num_games > 0:
        for name in loss_ratios:
            loss_ratios[name] /= num_games
            loss_ratios[name] *= 100
    return loss_ratios


def get_sorted_filenames_and_indices():
    files = [(int(f[5:]), f) for f in os.listdir(ppo_saved_models_dir) if f.find('model') != -1 and int(f[5:]) % 5000 == 0]
    files.sort(key=lambda x: x[0])
    return list(zip(*files))


def run_against_one_random(params_filename, games):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained1_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer")
        loss_ratios = do_test_games([player1, player2], games)
        return loss_ratios[player1.name] / 100


def run_against_3_randoms(params_filename, games):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained2_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer1")
        player3 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer2")
        player4 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer3")
        loss_ratios = do_test_games([player1, player2, player3, player4], games)
        return loss_ratios[player1.name] / 100


def run_against_3_defensives(params_filename, games):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained3_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = DefensivePlayer(DurakEnv.HAND_SIZE, "DefensivePlayer1")
        player3 = DefensivePlayer(DurakEnv.HAND_SIZE, "DefensivePlayer2")
        player4 = DefensivePlayer(DurakEnv.HAND_SIZE, "DefensivePlayer3")
        loss_ratios = do_test_games([player1, player2, player3, player4], games)
        return loss_ratios[player1.name] / 100


def run_against_3_aggressives(params_filename, games):
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained4_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = AggressivePlayer(DurakEnv.HAND_SIZE, "AggressivePlayer1")
        player3 = AggressivePlayer(DurakEnv.HAND_SIZE, "AggressivePlayer2")
        player4 = AggressivePlayer(DurakEnv.HAND_SIZE, "AggressivePlayer3")
        loss_ratios = do_test_games([player1, player2, player3, player4], games)
        return loss_ratios[player1.name] / 100


def plot_graph(title, indices, losses):
    plt.figure()
    plt.title(title)
    plt.xlabel("Player Model Number")
    plt.ylabel("Loss Ratios")
    plt.ylim(0, 1)
    plt.plot(indices, losses)


def train_ppo():
    """
    Trains a PPO player.
    :param sess: tensorflow session
    :param games_per_batch: number of training games per batch
    :param save_every: frequency of saving the model parameters (saved every save_every batches)
    :param training_games: total number of training games to train for
    """
    # Resets the folder of the saved parameters
    if not os.path.exists(ppo_saved_models_dir):
        os.mkdir(ppo_saved_models_dir)
    logging.basicConfig(filename='logs/PPOTrainer_log', level=logging.INFO)
    with tf.compat.v1.Session() as sess:
        trainer = PPOTrainer(sess,
                             games_per_batch=5,
                             training_steps_per_game=25,
                             learning_rate=0.00025,
                             clip_range=0.2,
                             save_every=100)
        trainer.train(500000)
    logging.shutdown()


def test_and_plot_ppo():
    games = 300
    logging.basicConfig(filename='logs/PPOTester_log', level=logging.INFO)
    indices, files = get_sorted_filenames_and_indices()
    # run against 1 random player, save results, and plot graph
    random_losses = []
    for f in files:
        random_losses.append(run_against_one_random(f, games))
    plot_graph("PPO models loss ratio against 1 Random Agent", indices, random_losses)

    # run against 3 random players, save results, and plot graph
    random_3_losses = []
    for f in files:
        random_3_losses.append(run_against_3_randoms(f, games))
    plot_graph("PPO models loss ratio against 3 Random Agents", indices, random_3_losses)

    # run against 3 defensive players, save results, and plot graph
    def_losses = []
    for f in files:
        def_losses.append(run_against_3_defensives(f, games))
    plot_graph("PPO models loss ratio against 3 Defensive Agents", indices, def_losses)

    # run against 3 aggressive players, save results, and plot graph
    agg_losses = []
    for f in files:
        agg_losses.append(run_against_3_aggressives(f, games))
    plot_graph("PPO models loss ratio against 3 Aggressive Agents", indices, agg_losses)

    logging.shutdown()


def generate_ppo_plots():
    train_ppo()
    test_and_plot_ppo()


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


def train_against_prev_iter(subdir, epochs=100, training_games_per_epoch=50):
    full_subdir_path = os.path.join(nfsp_saved_models_dir, subdir)
    if os.path.exists(full_subdir_path):
        shutil.rmtree(full_subdir_path)
    os.mkdir(full_subdir_path)
    learning_player = NFSPPlayer(hand_size, 'NFSP-PLAYER-1', 'cpu')
    learning_players = [learning_player]
    random1 = RandomPlayer(hand_size, 'random-1')
    training_players = [random1, RandomPlayer(hand_size, 'random-2')]
    trainer = NFSPTrainer(learning_players, training_players)
    cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    for epoch in range(epochs):
        print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
        print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
        trainer.train(training_games_per_epoch)
        learning_player.save_network(os.path.join(subdir, 'epoch-' + str(epoch + 1)))
        prev_iter1 = TrainedNFSPPlayer(hand_size, 'prev-iter1')
        prev_iter1.load_model(os.path.join(subdir, 'epoch-' + str(random.randint(max(1, epoch-10), epoch + 1))))
        training_players = [random1, prev_iter1]
        trainer = NFSPTrainer(learning_players, training_players)


def train_against_one_random(subdir, epochs=100, training_games_per_epoch=50):
    full_subdir_path = os.path.join(nfsp_saved_models_dir, subdir)
    if os.path.exists(full_subdir_path):
        shutil.rmtree(full_subdir_path)
    os.mkdir(full_subdir_path)
    learning_player = NFSPPlayer(hand_size, 'NFSP-PLAYER-1', 'cpu')
    learning_players = [learning_player]
    random1 = RandomPlayer(hand_size, 'random-1')
    training_players = [random1]
    trainer = NFSPTrainer(learning_players, training_players)
    cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    for epoch in range(epochs):
        print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
        print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
        trainer.train(training_games_per_epoch)
        learning_player.save_network(os.path.join(subdir, 'epoch-' + str(epoch + 1)))


def train_against_nfsp_agent(subdir, epochs=100, training_games_per_epoch=50):
    full_subdir_path = os.path.join(nfsp_saved_models_dir, subdir)
    if os.path.exists(full_subdir_path):
        shutil.rmtree(full_subdir_path)
    os.mkdir(full_subdir_path)
    learning_player1 = NFSPPlayer(hand_size, 'NFSP-PLAYER-1', 'cpu')
    learning_player2 = NFSPPlayer(hand_size, 'NFSP-PLAYER-2', 'cpu')
    learning_players = [learning_player1, learning_player2]
    trainer = NFSPTrainer(learning_players, [])
    cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    for epoch in range(epochs):
        print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
        print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
        trainer.train(training_games_per_epoch)
        learning_player1.save_network(os.path.join(subdir, 'epoch-' + str(epoch + 1)))


def graph(subdir, title, file_name, training_games_per_epoch=50, test_games_per_epoch_vs_test_players=500):
    filenames = os.listdir(os.path.join(nfsp_saved_models_dir, subdir))
    cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * len(filenames) + 1, training_games_per_epoch))
    loss_ratio_vs_1_random = []
    loss_ratio_vs_3_random = []
    trained = TrainedNFSPPlayer(hand_size, 'trained')
    random1 = RandomPlayer(hand_size, 'random-1')
    random2 = RandomPlayer(hand_size, 'random-2')
    random3 = RandomPlayer(hand_size, 'random-3')
    for filename in filenames:
        trained.load_model(os.path.join(subdir, filename))
        loss_ratios = do_test_games([trained, random1], test_games_per_epoch_vs_test_players)
        loss_ratio_vs_1_random.append(loss_ratios[trained.name])
        loss_ratios = do_test_games([trained, random1, random2, random3], test_games_per_epoch_vs_test_players)
        loss_ratio_vs_3_random.append(loss_ratios[trained.name])

    learning_player_info = {"VS. 1 random": (
        colors[1], loss_ratio_vs_1_random), "VS. 3 randoms": (
        colors[2], loss_ratio_vs_3_random)}
    plot(cumulative_training_games_per_epoch, learning_player_info,
         title, "Number of training games", "Loss Ratio", True)
    plt.savefig(file_name)


def train_and_plot_nfsp():
    epochs = 100
    training_games_per_epoch = 20
    test_games_per_epoch = 20
    subdir = 'train_against_prev_iter'
    train_against_prev_iter(subdir, epochs, training_games_per_epoch)
    graph(subdir, "NFSP trained against previous iterations", "Trained vs Previous Iterations.jpg", training_games_per_epoch, test_games_per_epoch)
    subdir = 'train_against_one_random'
    train_against_one_random(subdir, epochs, training_games_per_epoch)
    graph(subdir, "NFSP trained against random player", "Trained vs One Random Player.jpg", training_games_per_epoch, test_games_per_epoch)
    subdir = 'train_against_nfsp'
    train_against_nfsp_agent(subdir, epochs, training_games_per_epoch)
    graph(subdir, "NFSP trained against another NFSP", "Trained vs Another NFSP Player.jpg", training_games_per_epoch, )


def run_example_game():
    players = [HumanPlayer(hand_size, "Human"),
               DefensivePlayer(hand_size, "Defensive")]
    env = DurakEnv(players, True)
    done = False
    state = env.reset()
    env.render()
    while not done:
        act = (env.get_turn_player()).get_action(state, env.to_attack())
        state, _, done = env.step(act)
        env.render()
    env.end_gui()


def main():
    example = '0'
    ppo = '1'
    nfsp = '2'
    ppo_and_nfsp = '3'
    arg = sys.argv[1] if len(sys.argv) >= 2 else '0'
    show = False if len(sys.argv) < 3 or sys.argv[2] == 'f' else True
    if arg == example:
        print("Running example game")
        run_example_game()
    else:
        if show:
            print("Generated plots will be shown")
        else:
            print("Generated plots will not be shown")
        if arg in [ppo, ppo_and_nfsp]:
            print("Training and generating plots for the PPO player...")
            generate_ppo_plots()
        if arg in [nfsp, ppo_and_nfsp]:
            print("Training and generating plots for the NFSP player...")
            train_and_plot_nfsp()
        if show:
            plt.show()


if __name__ == '__main__':
    main()
