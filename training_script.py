from main import *
import random

def train_against_two_randoms(epochs=30, training_games_per_epoch=20, test_games_per_epoch_vs_test_players=50):
    # define players for learning phase
    learning_player1 = NFSPPlayer(hand_size, 'NFSP-PLAYER-1', 'cpu')
    learning_players = [learning_player1]
    training_players = [RandomPlayer(hand_size, 'random-1'), RandomPlayer(hand_size, 'random-2')]

    # define player for testing phase
    trained_player = TrainedNFSPPlayer(hand_size, 'Trained-NFSP')
    trained_player.load_from_other_player(learning_player1)
    test_players = [RandomPlayer(hand_size, "Random-3"), RandomPlayer(hand_size, 'Random-4'), trained_player]

    # build trainer
    trainer = NFSPTrainer(learning_players, training_players)

    cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    loss_ratio_vs_test_players = []
    loss_ratio_vs_training_players = []
    for epoch in range(epochs):
        print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
        print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
        print("Testing vs Training Players...")
        print("Testing vs Test Players...")
        trained_player.load_from_other_player(learning_player1)
        loss_ratios = do_test_games(test_players, test_games_per_epoch_vs_test_players)
        loss_ratio_vs_test_players.append(loss_ratios[trained_player.name])
        trainer.train(training_games_per_epoch)
        print('Testing vs Test Players loss ratio: ', loss_ratios)
        # save the network
        learning_player1.save_network('train_against_two_randoms/epoc-'+str(epoch+1))
    # learning_player_info = {"VS. Test Players": ((1., 0., 0.), loss_ratio_vs_test_players), "VS. Training Players": ((0., 1., 0.), loss_ratio_vs_training_players)}
    learning_player_info = {"VS. Test Players": ((1., 0., 0.), loss_ratio_vs_test_players)}
    plot(cumulative_training_games_per_epoch, learning_player_info, "Loss Ratio of Trained NFSP Player VS 2 Random Players", "Number of training games", "Loss Ratio", True)

    plt.savefig('train_against_two_randoms.jpg')


def train_against_prev_iter(epochs=25, training_games_per_epoch=100, test_games_per_epoch_vs_test_players=30):
    # define players for learning phase
    learning_player1 = NFSPPlayer(hand_size, 'NFSP-PLAYER-1', 'cpu')
    learning_players = [learning_player1]
    random1 = RandomPlayer(hand_size, 'random-1')
    training_players = [random1, RandomPlayer(hand_size, 'random-2')]

    # define player for testing phase
    trained_player = TrainedNFSPPlayer(hand_size, 'Trained-NFSP')
    trained_player.load_from_other_player(learning_player1)
    test_players = [RandomPlayer(hand_size, "Random-3"), DefensivePlayer(hand_size, 'Defensive-1'), trained_player]

    # build trainer
    trainer = NFSPTrainer(learning_players, training_players)

    cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
    loss_ratio_vs_test_players = []
    for epoch in range(epochs):
        print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
        print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
        trainer.train(training_games_per_epoch)
        print("Testing vs Training Players...")
        print("Testing vs Test Players...")
        trained_player.load_from_other_player(learning_player1)
        loss_ratios = do_test_games(test_players, test_games_per_epoch_vs_test_players)
        loss_ratio_vs_test_players.append(loss_ratios[trained_player.name])
        print('Testing vs Test Players loss ratio: ', loss_ratios)
        learning_player1.save_network('train_against_prev_iter/epoc-' + str(epoch + 1))

        prev_iter = TrainedNFSPPlayer(hand_size, 'prev-iter')
        prev_iter.load_model('train_against_prev_iter/epoc-'+str(random.randint(1, epoch + 1)))
        training_players = [random1, prev_iter]
        trainer = NFSPTrainer(learning_players, training_players)
        if epoch % 10 == 0:
            learning_player1 = NFSPPlayer(hand_size, 'NFSP-PLAYER-1', 'cpu')
            learning_player1.load_model('train_against_prev_iter/epoc-' + str(epoch + 1))


        # save the network
        learning_player1.save_network('train_against_two_randoms/epoc-'+str(epoch+1))
    # learning_player_info = {"VS. Test Players": ((1., 0., 0.), loss_ratio_vs_test_players), "VS. Training Players": ((0., 1., 0.), loss_ratio_vs_training_players)}
    learning_player_info = {"VS. Test Players": ((1., 0., 0.), loss_ratio_vs_test_players)}
    plot(cumulative_training_games_per_epoch, learning_player_info, "Loss Ratio of Trained NFSP Player VS 1 Random Players and 1 DefensivePlayer", "Number of training games", "Loss Ratio", True)

    learning_player1.save_network('train_against_two_randoms\9')
    plt.savefig('train_against_prev_iter.jpg')


# def train_against_self_k(epochs=50, training_games_per_epoch=20, test_games_per_epoch_vs_test_players=50):
#     # define players for learning phase
#     learning_player1 = NFSPPlayer(hand_size, 'NFSP-PLAYER-1', 'cpu')
#     learning_players = [learning_player1]
#     training_players = [RandomPlayer(hand_size, 'random-1'), RandomPlayer(hand_size, 'random-2')]
#
#     # define player for testing phase
#     trained_player = TrainedNFSPPlayer(hand_size, 'Trained-NFSP')
#     trained_player.load_from_other_player(learning_player1)
#     test_players = [RandomPlayer(hand_size, "Random-1"), RandomPlayer(hand_size, 'Random-2'), trained_player]
#
#     # build trainer
#     trainer = NFSPTrainer(learning_players, training_players)
#
#     cumulative_training_games_per_epoch = list(range(training_games_per_epoch, training_games_per_epoch * epochs + 1, training_games_per_epoch))
#     loss_ratio_vs_test_players = []
#     loss_ratio_vs_training_players = []
#     for epoch in range(epochs):
#         print("------------------------- Epoch", epoch + 1, "out of", epochs, "-------------------------")
#         print("Training for", training_games_per_epoch, "games (total number of training games: " + str(cumulative_training_games_per_epoch[epoch]) + ")")
#         trainer.train(training_games_per_epoch)
#         print("Testing vs Training Players...")
#         loss_ratios = do_test_games(test_players, test_games_per_epoch_vs_test_players)
#         # loss_ratios = do_test_games(trainer.get_all_players(), test_games_per_epoch_vs_test_players)
#         loss_ratio_vs_training_players.append(loss_ratios[trained_player.name])
#         print('Testing vs Training Players loss ratio: ', loss_ratios)
#         print("Testing vs Test Players...")
#         trained_player.load_from_other_player(learning_player1)
#         loss_ratios = do_test_games(test_players, test_games_per_epoch_vs_test_players)
#         loss_ratio_vs_test_players.append(loss_ratios[trained_player.name])
#         print('Testing vs Test Players loss ratio: ', loss_ratios)
#         # save the network
#         learning_player1.save_network('train_against_two_randoms/epoc-'+str(epoch+1))
#     # learning_player_info = {"VS. Test Players": ((1., 0., 0.), loss_ratio_vs_test_players), "VS. Training Players": ((0., 1., 0.), loss_ratio_vs_training_players)}
#     learning_player_info = {"VS. Test Players": ((1., 0., 0.), loss_ratio_vs_test_players)}
#     plot(cumulative_training_games_per_epoch, learning_player_info, "Loss Ratio of Trained NFSP Player VS 2 Random Players", "Number of training games", "Loss Ratio", True)
#
#     learning_player1.save_network('train_against_two_randoms\9')
#     plt.savefig('new_try6.jpg')


if __name__ == '__main__':
    # train_against_two_randoms()
    train_against_prev_iter()


