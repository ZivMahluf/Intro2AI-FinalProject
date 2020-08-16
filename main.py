from NFSPTrainedPlayer import TrainedNFSPPlayer
from AggressivePlayer import AggressivePlayer
from DefensivePlayer import DefensivePlayer
from RandomPlayer import RandomPlayer
from HumanPlayer import HumanPlayer
from DurakPlayer import DurakPlayer
from NFSPPlayer import NFSPPlayer
from PPOPlayer import from_path

from Types import List, Dict, NumberType, Tuple
from NFSPTrainer import NFSPTrainer
from PPOTrainer import PPOTrainer
from DurakEnv import DurakEnv

import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import random
import shutil
import sys
import os


"""
General useful parameters.
"""
colors = ["gray", "yellowgreen", "blue", "brown", "deeppink", "red", "coral", "peru", "darkorange", "gold", "darkkhaki",
          "yellow", "forestgreen", "lime", "darkslategray", "cyan", "dodgerblue", "royalblue", "indigo", "purple"]
hand_size = DurakEnv.HAND_SIZE
nfsp_saved_models_dir = os.path.join(os.getcwd(), "NFSP-models")
ppo_saved_models_dir = os.path.join(os.getcwd(), "PPOParams")


def do_test_games(players: List[DurakPlayer], num_games: int, steps_lim=500) -> Dict[str, float]:
    """
    Performs test games and collects loss ratios in % of each player.
    :param players: list of players.
    :param num_games: number of games.
    :param steps_lim: limit of steps for each game.
    :return: dictionary mapping the name of each player to their % of losses
    """
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
    """
    Loads every 5000'th model fro mthe directory in which the PPO models are saved.
    :return: Sorted list of files of PPO models from the directory in which they are saved.
    """
    files = [(int(f[5:]), f) for f in os.listdir(ppo_saved_models_dir) if f.find('model') != -1 and int(f[5:]) % 5000 == 0]
    files.sort(key=lambda x: x[0])
    return list(zip(*files))


def run_against_one_random(params_filename, games):
    """
    Runs test games against a single random player and returns the loss ration in range [0, 1]
    :param params_filename: path of the file containing the parameters of the model to test.
    :param games: number of test games.
    :return: loss ratio of the player in range [0, 1]
    """
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained1_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer")
        loss_ratios = do_test_games([player1, player2], games)
        return loss_ratios[player1.name] / 100


def run_against_3_randoms(params_filename, games):
    """
    Runs test games against 3 random players and returns the loss ration in range [0, 1]
    :param params_filename: path of the file containing the parameters of the model to test.
    :param games: number of test games.
    :return: loss ratio of the player in range [0, 1]
    """
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained2_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer1")
        player3 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer2")
        player4 = RandomPlayer(DurakEnv.HAND_SIZE, "RandomPlayer3")
        loss_ratios = do_test_games([player1, player2, player3, player4], games)
        return loss_ratios[player1.name] / 100


def run_against_3_defensives(params_filename, games):
    """
    Runs test games against 3 defensive players and returns the loss ration in range [0, 1]
    :param params_filename: path of the file containing the parameters of the model to test.
    :param games: number of test games.
    :return: loss ratio of the player in range [0, 1]
    """
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained3_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = DefensivePlayer(DurakEnv.HAND_SIZE, "DefensivePlayer1")
        player3 = DefensivePlayer(DurakEnv.HAND_SIZE, "DefensivePlayer2")
        player4 = DefensivePlayer(DurakEnv.HAND_SIZE, "DefensivePlayer3")
        loss_ratios = do_test_games([player1, player2, player3, player4], games)
        return loss_ratios[player1.name] / 100


def run_against_3_aggressives(params_filename, games):
    """
    Runs test games against 3 aggressive players and returns the loss ration in range [0, 1]
    :param params_filename: path of the file containing the parameters of the model to test.
    :param games: number of test games.
    :return: loss ratio of the player in range [0, 1]
    """
    with tf.compat.v1.Session() as sess:
        player1 = from_path(sess, DurakEnv.HAND_SIZE, "Trained4_PPO_" + params_filename, os.path.join(ppo_saved_models_dir,  params_filename))
        player2 = AggressivePlayer(DurakEnv.HAND_SIZE, "AggressivePlayer1")
        player3 = AggressivePlayer(DurakEnv.HAND_SIZE, "AggressivePlayer2")
        player4 = AggressivePlayer(DurakEnv.HAND_SIZE, "AggressivePlayer3")
        loss_ratios = do_test_games([player1, player2, player3, player4], games)
        return loss_ratios[player1.name] / 100


def plot_graph(title, x_axis, y_axis):
    """
    Plots a single graph on a figure.
    :param title: title of the plot.
    :param x_axis: x axis values.
    :param y_axis: y axis values.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Player Model Number")
    plt.ylabel("Loss Ratios")
    plt.ylim(0, 1)
    plt.plot(x_axis, y_axis)


def train_ppo():
    """
    Trains a PPO player.
    """
    # Resets the folder of the saved parameters
    if not os.path.exists(ppo_saved_models_dir):
        os.mkdir(ppo_saved_models_dir)
    with tf.compat.v1.Session() as sess:
        trainer = PPOTrainer(sess,
                             games_per_batch=5,
                             training_steps_per_game=25,
                             learning_rate=0.00025,
                             clip_range=0.2,
                             save_every=100)
        trainer.train(500000)


def test_and_plot_ppo():
    """
    Tests the PPO player against test cases.
    """
    games = 100
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
    """
    Calls the training function for the PPO player and then calls the test and plot function.
    """
    train_ppo()
    test_and_plot_ppo()


def plot(x_axis: List[NumberType], y_axes: Dict[str, Tuple[Tuple[float, float, float], List[NumberType]]], title: str, x_label: str, y_label: str, legend: bool):
    """
    Plots multiple plots to the same figure.
    :param x_axis: values of the x axis.
    :param y_axes: a dictionary mapping the label of the plots to the color of the plot and the y axis values.
    :param title: figure title.
    :param x_label: x axis label.
    :param y_label: y axis label.
    :param legend: weather to display a legend (mapping each plot label to the plot)
    """
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
    """
    Trains a NFSP agent against a previous iteration of itself.
    :param subdir: subdirectory in which to save the models while training.
    :param epochs: number of training epochs.
    :param training_games_per_epoch: number of training games per epoch.
    """
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
    """
    Trains a NFSP agent against a random player.
    :param subdir: subdirectory in which to save the models while training.
    :param epochs: number of training epochs.
    :param training_games_per_epoch: number of training games per epoch.
    """
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
    """
    Trains two NFSP agents against each other.
    :param subdir: subdirectory in which to save the models while training.
    :param epochs: number of training epochs.
    :param training_games_per_epoch: number of training games per epoch.
    """
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
    """
    Plots the results of the NFSP models saved in the given subdirectory vs. test cases.
    :param subdir: subdirectory in which the NFSP models are saved.
    :param title: figure title.
    :param file_name: name of file to save the figure.
    :param training_games_per_epoch: number of training games per epoch used in training of the models.
    :param test_games_per_epoch_vs_test_players: number of test games for each model against each test case.
    """
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
    """
    Calls the training and plotting functions for the NFSP agents.
    """
    epochs = 100
    training_games_per_epoch = 100
    test_games_per_epoch = 100
    subdir = 'train_against_prev_iter'
    train_against_prev_iter(subdir, epochs, training_games_per_epoch)
    graph(subdir, "NFSP trained against previous iterations", "Trained vs Previous Iterations.jpg", training_games_per_epoch, test_games_per_epoch)
    subdir = 'train_against_one_random'
    train_against_one_random(subdir, epochs, training_games_per_epoch)
    graph(subdir, "NFSP trained against random player", "Trained vs One Random Player.jpg", training_games_per_epoch, test_games_per_epoch)
    subdir = 'train_against_nfsp'
    train_against_nfsp_agent(subdir, epochs, training_games_per_epoch)
    graph(subdir, "NFSP trained against another NFSP", "Trained vs Another NFSP Player.jpg", training_games_per_epoch, test_games_per_epoch)


def run_example_game():
    """
    Runs a complete example game.
    """
    players = [HumanPlayer(hand_size, "Human"),
               DefensivePlayer(hand_size, "Player")]
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
    """
    Runs the appropriate function according to the given arguments.
    """
    example = '0'
    ppo = '1'
    nfsp = '2'
    ppo_and_nfsp = '3'
    arg = sys.argv[1] if len(sys.argv) >= 2 else '0'
    show = False if len(sys.argv) >= 3 and sys.argv[2] == 'f' else True
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
