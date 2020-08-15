from NFSPTrainedPlayer import TrainedNFSPPlayer
from AggressivePlayer import AggressivePlayer
from DefensivePlayer import DefensivePlayer
from RandomPlayer import RandomPlayer
from HumanPlayer import HumanPlayer
from DurakPlayer import DurakPlayer
from NFSPPlayer import NFSPPlayer
from PPOPlayer import PPOPlayer

from Types import List
from NFSPTrainer import NFSPTrainer
from PPOTrainer import PPOTrainer
from PPONetwork import PPONetwork
from DurakEnv import DurakEnv

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
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


def do_test_games(players: List[DurakPlayer], num_games: int):
    loss_ratios = {player.name: 0 for player in players}
    env = DurakEnv(players, False)
    print_every = min(num_games // 10, 50)
    for i in range(1, num_games + 1):
        if num_games >= 100:
            if i == 1 or (i % print_every) == 0:
                print("playing test game", i, "out of", num_games)
        state = env.reset()
        done = False
        while not done:
            act = (env.get_turn_player()).get_action(state, env.to_attack())
            state, _, done = env.step(act)
        loser = env.get_loser()
        if loser is not None:
            loss_ratios[loser.name] += 1
    if num_games > 0:
        for name in loss_ratios:
            loss_ratios[name] /= num_games
            loss_ratios[name] *= 100
    return loss_ratios


def train_ppo(sess, games_per_batch, save_every, training_games):
    """
    Trains a PPO player.
    :param sess: tensorflow session
    :param games_per_batch: number of training games per batch
    :param save_every: frequency of saving the model parameters (saved every save_every batches)
    :param training_games: total number of training games to train for
    """
    # Resets the folder of the saved parameters (uncomment to enable reset directory)
    # if os.path.exists(ppo_saved_models_dir):
    #     shutil.rmtree(ppo_saved_models_dir)
    # os.mkdir(ppo_saved_models_dir)
    trainer = PPOTrainer(sess,
                         games_per_batch=games_per_batch,
                         training_steps_per_game=25,
                         learning_rate=0.00025,
                         clip_range=0.2,
                         save_every=save_every)
    name = os.path.join(ppo_saved_models_dir, "model0")
    trainer.trainingNetwork.saveParams(name)
    trainer.train(training_games)


def plot_trained_ppo_player_vs_test_cases(sess, title, test_games_per_model=20):
    """
    Trains a PPO player for a given number of games.
    After training, loads the saved models and tests them for the given number of games against the given test players.
    The % of losses of each trained PPO player are plotted against the number of training games up to the point of saving the tested model (meaning we
    plot the loss % of the player against the test players as a function of the training games).
    :param sess: tensorflow session
    :param title: title for the plot
    :param test_games_per_model: how many games each model plays against the test players
    """
    model_names = os.listdir(ppo_saved_models_dir)
    if len(model_names) > 1:
        prev_network = PPONetwork(sess, PPOPlayer.input_dim, PPOPlayer.output_dim, "Prev_Network")
        prev_player = PPOPlayer(hand_size, "Prev_Player", prev_network, sess)
        prev_player.test_phase = True
        curr_network = PPONetwork(sess, PPOPlayer.input_dim, PPOPlayer.output_dim, "Curr_Network")
        parameters = joblib.load(os.path.join(ppo_saved_models_dir, model_names[0]))
        curr_network.loadParams(parameters)
        curr_player = PPOPlayer(hand_size, "Curr_Player", curr_network, sess)
        curr_player.test_phase = True
        training_player_1 = PPOPlayer(hand_size, "Training_Player_1", curr_network, sess)
        training_player_2 = PPOPlayer(hand_size, "Training_Player_2", curr_network, sess)
        training_player_3 = PPOPlayer(hand_size, "Training_Player_3", curr_network, sess)
        test_players_lists = {"vs. Training Players": [training_player_1, training_player_2, training_player_3],
                              "vs. 1 Defensive Player": [DefensivePlayer(hand_size, "Defensive Test Player")],
                              "vs. 1 Aggressive Player": [AggressivePlayer(hand_size, "Aggressive Test Player")],
                              "vs. 1 Random Player": [RandomPlayer(hand_size, "Random Test Player")],
                              "vs. Previous Version": [prev_player]}
        loss_ratios_per_test_list = {key: list() for key in test_players_lists.keys()}
        for i in range(1, len(model_names)):
            print("Testing model", i, "out of", len(model_names) - 1)
            prev_network.loadParams(parameters)
            parameters = joblib.load(os.path.join(ppo_saved_models_dir, model_names[i]))
            curr_network.loadParams(parameters)
            for key in test_players_lists:
                print("Testing " + key + " ...")
                loss_ratios = do_test_games([curr_player] + test_players_lists[key], test_games_per_model)
                loss_ratios_per_test_list[key].append(loss_ratios[curr_player.name])
        plt.figure(figsize=figure_size)
        plt.title(title)
        plt.xlabel("Player Model Number")
        plt.ylabel("Loss Ratio in %")
        i = 0
        for key in loss_ratios_per_test_list:
            plt.plot(loss_ratios_per_test_list[key], color=colors[i], label=key)
            i += 1
        plt.legend(loc=figure_legend_loc)
        plt.savefig(title.replace('\n', '') + ".png")


def generate_ppo_plots():
    games_per_batch = 20
    save_every = 5
    num_models = 100  # number of models to save and test
    test_games_per_saved_model = 50
    with tf.compat.v1.Session() as sess:
        train_ppo(sess, games_per_batch=games_per_batch, save_every=save_every, training_games=games_per_batch * save_every * num_models)
        plot_trained_ppo_player_vs_test_cases(sess, "Loss Ratios of PPO Player", test_games_per_saved_model)


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
