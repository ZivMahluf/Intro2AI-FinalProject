from NFSPTrainer import NFSPTrainer
from PPOTrainer import PPOTrainer
import tensorflow as tf
import matplotlib.pyplot as plt


def train_PPO(epochs=2, games_per_epoch=5, test_games_per_epoch=10):
    with tf.compat.v1.Session() as sess:
        trainer = PPOTrainer(sess, games_per_batch=5, training_steps_per_game=25, learning_rate=0.00025, clip_range=0.2, save_every=2)
        ratios = {name: list() for name in trainer.get_learning_players_names()}
        for epoch in range(epochs):
            trainer.train(games_per_epoch)
            loss_ratios = trainer.test(test_games_per_epoch)
            for name in loss_ratios:
                ratios[name].append(loss_ratios[name])
    return ratios, trainer.get_players()


def train_NFSP(epochs=2, games_per_epoch=5, test_games_per_epoch=10):
    trainer = NFSPTrainer()
    ratios = {name: list() for name in trainer.get_learning_players_names()}
    for epoch in range(epochs):
        trainer.train(games_per_epoch)
        loss_ratios = trainer.test(test_games_per_epoch)
        for name in loss_ratios:
            ratios[name].append(loss_ratios[name])
    return ratios, trainer.get_learning_players()


def main():
    ratios, players = train_PPO()
    for player in players:
        name = player.name
        plt.figure()
        plt.title(name)
        plt.plot(ratios[name])

    # ratios, players = train_NFSP()
    # for player in players:
    #     name = player.name
    #     plt.figure()
    #     plt.title(name)
    #     plt.plot(ratios[name])

    plt.show()


if __name__ == '__main__':
    main()
