from AggressivePlayer import AggressivePlayer
from DefensivePlayer import DefensivePlayer
from HumanPlayer import HumanPlayer
from RandomPlayer import RandomPlayer
from NFSPPlayer import NFSPPlayer
from NFSPTrainer import NFSPTrainer
from PPOTrainer import PPOTrainer
from DurakEnv import DurakEnv
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Tuple


Numeric = Union[int, float]
Color = Tuple[float, float, float]
hand_size = DurakEnv.HAND_SIZE


def do_test_games(players, num_games):
    loss_ratios = {player.name: 0 for player in players}
    env = DurakEnv(players, False)
    print_every = min(num_games // 10, 50)
    for i in range(1, num_games + 1):
        state = env.reset()
        done = False
        while not done:
            act = (env.get_turn_player()).get_action(state, env.to_attack())
            state, _, done, _ = env.step(act)
        loser = env.get_loser()
        if loser is not None:
            loss_ratios[loser.name] += 1
        if num_games >= 20:
            if i == 1 or (i % print_every) == 0:
                print("test game", i, "out of", num_games)
    for name in loss_ratios:
        if num_games > 0:
            loss_ratios[name] /= num_games
    return loss_ratios


def train_PPO(session, epochs=2, games_per_epoch=5, test_games_per_epoch=10):
    trainer = PPOTrainer(session, games_per_batch=5, training_steps_per_game=25, learning_rate=0.00025, clip_range=0.2, save_every=2)
    ratios = {name: list() for name in trainer.get_learning_players_names()}
    for epoch in range(epochs):
        print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------") if epochs > 1 else None
        trainer.train(games_per_epoch)
        loss_ratios = do_test_games(trainer.get_players(), test_games_per_epoch)
        for name in loss_ratios:
            ratios[name].append(loss_ratios[name])
    return ratios, trainer.get_players()


def train_NFSP(epochs=2, games_per_epoch=5, test_games_per_epoch=10):
    learning_players = [NFSPPlayer(hand_size, "NFSP-1"), NFSPPlayer(hand_size, "NFSP-2")]
    other_players = [RandomPlayer(hand_size, "Random-1"), AggressivePlayer(hand_size, "Aggressive-1")]
    trainer = NFSPTrainer(learning_players, other_players)
    ratios = {name: list() for name in trainer.get_learning_players_names()}
    for epoch in range(epochs):
        print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------") if epochs > 1 else None
        trainer.train(games_per_epoch)
        loss_ratios = do_test_games(trainer.get_all_players(), test_games_per_epoch)
        for name in loss_ratios:
            ratios[name].append(loss_ratios[name])
    return ratios, learning_players


def plot(x_axis: List[Numeric], y_axes: Dict[str, Tuple[Color, List[Numeric]]], title: str, x_label: str, y_label: str, legend: bool):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for plot_label in y_axes:
        color, values = y_axes[plot_label]
        plt.plot(x_axis, values, color=color, label=plot_label)
    if legend:
        plt.legend()


def run_example_game():
    players = [HumanPlayer(hand_size, "Human"), DefensivePlayer(hand_size, "Defensive")]
    env = DurakEnv(players, True)
    done = False
    state = env.reset()
    env.render()
    while not done:
        act = (env.get_turn_player()).get_action(state, env.to_attack())
        state, _, done, _ = env.step(act)
        env.render()


def main():
    # Example game:
    run_example_game()

    # # Example PPO training and testing against random player with plots:
    # with tf.compat.v1.Session() as sess:
    #     # training the PPO players and plotting their loss ratio for each epoch
    #     epochs = 50
    #     training_games_per_epoch = 5
    #     test_games_per_epoch = 20
    #     loss_ratios, ppo_players = train_PPO(sess, epochs, training_games_per_epoch, test_games_per_epoch)
    #     players_information = {name: ((2 * i / (3 * len(ppo_players)), 3 * i / (5 * len(ppo_players)), i / len(ppo_players)), loss_ratios[name]) for i, name in enumerate(loss_ratios.keys())}
    #     x_axis = list(range(epochs))
    #     plot(x_axis, players_information, "Loss Ratios of the PPO Players Against Each Other", "Epoch", "Loss Ratio", True)
    #     # testing one of the PPO players against a random player
    #     players = [ppo_players[0], RandomPlayer(hand_size, "Random 1"), RandomPlayer(hand_size, "Random 2"), AggressivePlayer(hand_size, "Aggressive"), DefensivePlayer(hand_size, "Defensive")]
    #     test_games = 1000
    #     ratios = do_test_games(players, test_games)
    #     print("Trained PPO player won " + str(100 - ratios[ppo_players[0].name] * 100) + "% of", test_games, "games against 2 random players, an aggressive player, and a defensive player")
    #     plt.show()

    # # Example of plotting the loss ratio of a trained PPO player vs 3 random players as a function of the number of training games:
    # with tf.compat.v1.Session() as sess:
    #     trainer = PPOTrainer(sess, games_per_batch=5, training_steps_per_game=25, learning_rate=0.00025, clip_range=0.2, save_every=2)
    #
    #     training_games_per_epoch = 100
    #     epochs = 50
    #     cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    #     test_games_per_epoch_vs_random_players = 100
    #
    #     opponent1 = RandomPlayer(hand_size, "Random1")
    #     opponent2 = RandomPlayer(hand_size, "Random2")
    #     opponent3 = RandomPlayer(hand_size, "Random3")
    #
    #     loss_ratio_vs_random_opponents = []
    #
    #     for epoch in range(epochs):
    #         print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
    #
    #         print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
    #         trainer.train(training_games_per_epoch)
    #
    #         print("Testing...")
    #         sample_player = trainer.get_players()[0]
    #         players_for_test = [sample_player, opponent1, opponent2, opponent3]
    #         loss_ratios = do_test_games(players_for_test, test_games_per_epoch_vs_random_players)
    #         loss_ratio_vs_random_opponents.append(loss_ratios[sample_player.name])
    #
    #     ppo_player_info = {"PPO Player": ((0., 0., 1.), loss_ratio_vs_random_opponents)}
    #     plot(cumulative_training_games_per_epoch, ppo_player_info, "Loss Ratio of Trained PPO Player VS 3 Random Players", "Number of training games", "Loss Ratio", False)
    #     plt.show()

    # # Example Plot:
    # players = [AggressivePlayer(hand_size, "Aggressive Player"), DefensivePlayer(hand_size, "Defensive Player"), RandomPlayer(hand_size, "Random Player")]
    # players_information = {player.name: ((i / len(players), i / len(players), i / len(players)), []) for i, player in enumerate(players)}
    # epochs = 50
    # games = 100
    # for epoch in range(epochs):
    #     ratios = do_test_games(players, games)
    #     for name in ratios:
    #         players_information[name][1].append(ratios[name])
    #     print('epoch', epoch + 1)
    # plot(list(range(epochs)), players_information, "Example Plot", "Epoch", "Loss ratio", True)
    # plt.show()

    # # Example of plotting the loss ratio of a trained NFSP Player (trained against 3 random players) vs. 1 Aggressive, 1 Defensive, and 1 Random player
    # # as a function of the number of training games:
    # learning_player = NFSPPlayer(hand_size, "NFSP Player", 'cpu')
    # learning_players = [learning_player]
    # training_players = [RandomPlayer(hand_size, "Random 1"), RandomPlayer(hand_size, "Random 2"), RandomPlayer(hand_size, "Random 3")]
    # test_players = [RandomPlayer(hand_size, "Random"), AggressivePlayer(hand_size, "Aggressive"), DefensivePlayer(hand_size, "Defensive"), learning_player]
    # trainer = NFSPTrainer(learning_players, training_players)
    # training_games_per_epoch = 100
    # epochs = 25
    # cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    # test_games_per_epoch_vs_test_players = 50
    # loss_ratio_vs_test_players = []
    # loss_ratio_vs_training_players = []
    # for epoch in range(epochs):
    #     print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
    #     print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
    #     trainer.train(training_games_per_epoch)
    #     print("Testing vs Training Players...")
    #     loss_ratios = do_test_games(trainer.get_all_players(), test_games_per_epoch_vs_test_players)
    #     loss_ratio_vs_training_players.append(loss_ratios[learning_player.name])
    #     print("Testing vs Test Players...")
    #     loss_ratios = do_test_games(test_players, test_games_per_epoch_vs_test_players)
    #     loss_ratio_vs_test_players.append(loss_ratios[learning_player.name])
    # learning_player_info = {"VS. Test Players": ((1., 0., 0.), loss_ratio_vs_test_players),
    #                         "VS. Training Players": ((0., 1., 0.), loss_ratio_vs_training_players)}
    # plot(cumulative_training_games_per_epoch, learning_player_info, "Loss Ratio of Trained NFSP Player VS 3 Random Players", "Number of training games", "Loss Ratio", True)
    # plt.show()


if __name__ == '__main__':
    main()
