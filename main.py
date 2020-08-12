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
import shutil
import joblib
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
    # Resets the folder of the saved parameters
    if os.path.exists(ppo_saved_models_dir):
        shutil.rmtree(ppo_saved_models_dir)
    os.mkdir(ppo_saved_models_dir)
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
    test_games = 50
    with tf.compat.v1.Session() as sess:
        train_ppo(sess, games_per_batch=games_per_batch, save_every=save_every, training_games=games_per_batch * save_every * num_models)
        plot_trained_ppo_player_vs_test_cases(sess, "Loss Ratios of PPO Player", test_games)


def plot_train_and_test_nfsp(training_players, epochs, games_per_epoch, test_games_per_epoch, subdir, title):
    if os.path.exists(os.path.join(nfsp_saved_models_dir, subdir)):
        shutil.rmtree(os.path.join(nfsp_saved_models_dir, subdir))
    os.mkdir(os.path.join(nfsp_saved_models_dir, subdir))
    test_players_lists = {"vs. Training Players": training_players,
                          "vs. 1 Defensive Player": [DefensivePlayer(hand_size, "Defensive Test Player")],
                          "vs. 1 Aggressive Player": [AggressivePlayer(hand_size, "Aggressive Test Player")],
                          "vs. 1 Random Player": [RandomPlayer(hand_size, "Random Test Player")]}
    loss_ratios_per_test_list = {key: list() for key in test_players_lists.keys()}
    learning_player = NFSPPlayer(hand_size, "Learning Player")
    base_name = os.path.join(subdir, "Model")
    trainer = NFSPTrainer([learning_player], training_players)
    x_axis = [0]
    print("Pre-Training Testing...")
    for key in test_players_lists:
        print("Testing " + key + " ...")
        loss_ratios = do_test_games(test_players_lists[key] + [learning_player], test_games_per_epoch)
        loss_ratios_per_test_list[key].append(loss_ratios[learning_player.name])
    for epoch in range(1, epochs + 1):
        print("Epoch", epoch, "out of", epochs)
        print("Training...")
        trainer.train(games_per_epoch)
        x_axis.append(epoch * games_per_epoch)
        for key in test_players_lists:
            print("Testing " + key + " ...")
            loss_ratios = do_test_games(test_players_lists[key] + [learning_player], test_games_per_epoch)
            loss_ratios_per_test_list[key].append(loss_ratios[learning_player.name])
        learning_player.save_network(base_name + str(epoch))
    plt.figure(figsize=figure_size)
    plt.title(title)
    plt.xlabel("Number of Training Games")
    plt.ylabel("Loss Ratio in %")
    i = 0
    for key in loss_ratios_per_test_list:
        plt.plot(x_axis, loss_ratios_per_test_list[key], color=colors[i], label=key)
        i += 1
    plt.legend(loc=figure_legend_loc)
    plt.savefig(title.replace('\n', '') + ".png")


def plot_test_nsfp_vs_prev_version(subdir, test_games_per_model, title):
    model_names = os.listdir(os.path.join(nfsp_saved_models_dir, subdir))
    num_models = len(model_names)
    if num_models > 1:
        curr_model = TrainedNFSPPlayer(hand_size, "Current Model")
        curr_model.load_model(os.path.join(subdir, model_names[0]))
        prev_model = TrainedNFSPPlayer(hand_size, "Previous Model")  # starts as a random model
        loss_ratios_vs_prev_model = []
        print("Testing model 1 against untrained player ...")
        loss_ratios = do_test_games([curr_model, prev_model], test_games_per_model)
        loss_ratios_vs_prev_model.append(loss_ratios[curr_model.name])
        i = 1
        for model_name in model_names[1:]:
            print("Testing model", i, "against model", i - 1, "...")
            i += 1
            prev_model.load_from_other_player(curr_model)
            curr_model.load_model(os.path.join(subdir, model_name))
            loss_ratios = do_test_games([curr_model, prev_model], test_games_per_model)
            loss_ratios_vs_prev_model.append(loss_ratios[curr_model.name])
        plt.figure(figsize=figure_size)
        plt.title(title)
        plt.xlabel("Current Model Number")
        plt.ylabel("Loss Ratio in % Against Previous Model")
        plt.plot(loss_ratios_vs_prev_model, color=colors[12])
        plt.savefig(title.replace('\n', '') + ".png")


def plot_train_and_test_nfsp_vs_prev_version(epochs, games_per_epoch, test_games_per_epoch, subdir, title):
    if os.path.exists(os.path.join(nfsp_saved_models_dir, subdir)):
        shutil.rmtree(os.path.join(nfsp_saved_models_dir, subdir))
    os.mkdir(os.path.join(nfsp_saved_models_dir, subdir))
    learning_player = NFSPPlayer(hand_size, "Learning Player")
    training_player = NFSPPlayer(hand_size, "Training Player")
    test_players_lists = {"vs. Training Players": [training_player],
                          "vs. 1 Defensive Player": [DefensivePlayer(hand_size, "Defensive Test Player")],
                          "vs. 1 Aggressive Player": [AggressivePlayer(hand_size, "Aggressive Test Player")],
                          "vs. 1 Random Player": [RandomPlayer(hand_size, "Random Test Player")]}
    loss_ratios_per_test_list = {key: list() for key in test_players_lists.keys()}
    x_axis = [0]
    print("Pre-Training Testing...")
    for key in test_players_lists:
        print("Testing " + key + " ...")
        loss_ratios = do_test_games(test_players_lists[key] + [learning_player], test_games_per_epoch)
        loss_ratios_per_test_list[key].append(loss_ratios[learning_player.name])
    trainer = NFSPTrainer([learning_player], [training_player])
    for epoch in range(1, epochs + 1):
        print("Epoch", epoch, "out of", epochs)
        print("Training...")
        trainer.train(games_per_epoch)
        x_axis.append(epoch * games_per_epoch)
        for key in test_players_lists:
            print("Testing " + key + " ...")
            loss_ratios = do_test_games(test_players_lists[key] + [learning_player], test_games_per_epoch)
            loss_ratios_per_test_list[key].append(loss_ratios[learning_player.name])
        training_player.load_network_from_other_by_reference(learning_player)
    plt.figure(figsize=figure_size)
    plt.title(title)
    plt.xlabel("Number of Training Games")
    plt.ylabel("Loss Ratio in %")
    i = 0
    for key in loss_ratios_per_test_list:
        plt.plot(x_axis, loss_ratios_per_test_list[key], color=colors[i], label=key)
        i += 1
    plt.legend(loc=figure_legend_loc)
    # plt.savefig(title.replace('\n', '') + ".png")


def generate_nfsp_plots():
    games_per_batch = 25
    num_models = 100  # number of models to save and test, also the number of epochs for the NFSP player
    test_games = 20
    random_1 = RandomPlayer(hand_size, "Random 1")
    random_2 = RandomPlayer(hand_size, "Random 2")
    defensive_1 = DefensivePlayer(hand_size, "Defensive 1")
    training_players_lists = [[random_1, random_2],
                              [random_1, random_2, RandomPlayer(hand_size, "Random 3")],
                              [defensive_1],
                              [defensive_1, DefensivePlayer(hand_size, "Defensive 2"), DefensivePlayer(hand_size, "Defensive 3")]]
    subdirectories = ["2-Random", "3-Random", "1-Defensive", "3-Defensive"]
    train_test_cases_titles = ["Loss Ratios of NFSP Player Trained With 2 Random Players",
                               "Loss Ratios of NFSP Player Trained With 3 Random Players",
                               "Loss Ratios of NFSP Player Trained With 1 Defensive Player",
                               "Loss Ratios of NFSP Player Trained With 3 Defensive Players"]
    test_vs_prev_version_titles = ["Loss Ratio of Each Model Against Previous Model \nTrained With 2 Random Players",
                                   "Loss Ratio of Each Model Against Previous Model \nTrained With 3 Random Players",
                                   "Loss Ratio of Each Model Against Previous Model \nTrained With 1 Defensive Player",
                                   "Loss Ratio of Each Model Against Previous Model \nTrained With 3 Defensive Players"]
    for training_players, subdirectory, train_test_case_title, test_vs_prev_version_title in zip(training_players_lists, subdirectories, train_test_cases_titles, test_vs_prev_version_titles):
        plot_train_and_test_nfsp(training_players, num_models, games_per_batch, test_games, subdirectory, train_test_case_title)
        plot_test_nsfp_vs_prev_version(subdirectory, test_games, test_vs_prev_version_title)
    plot_train_and_test_nfsp_vs_prev_version(num_models, games_per_batch, test_games, "Prev-Version",
                                             "Loss Ratios of NFSP Player Trained Against the Previous Version of Itself")


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


def main():
    # run_example_game()
    # generate_ppo_plots()
    generate_nfsp_plots()
    # plt.show()


if __name__ == '__main__':
    main()
