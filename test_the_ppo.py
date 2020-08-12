import numpy as np
from PPONetwork import PPONetwork, PPOModel
import tensorflow as tf
import joblib
from Deck import Deck
from DurakEnv import DurakEnv
from PPOPlayer import PPOPlayer
from RandomPlayer import RandomPlayer
from DefensivePlayer import DefensivePlayer
import logging
import time
from PPOTrainer import PPOTrainer

if __name__ == "__main__":
    logging.basicConfig(filename='logs/PPOTester_log', level=logging.INFO)
    with tf.compat.v1.Session() as sess:
        num_games = 100
        player1 = PPOPlayer(DurakEnv.HAND_SIZE, "PPO Player", None, sess)
        player2 = DefensivePlayer(DurakEnv.HAND_SIZE, "def 1")
        player3 = DefensivePlayer(DurakEnv.HAND_SIZE, "def 2")
        player4 = DefensivePlayer(DurakEnv.HAND_SIZE, "def 3")
        # player2 = RandomPlayer(DurakEnv.HAND_SIZE, "random 1")
        # player3 = RandomPlayer(DurakEnv.HAND_SIZE, "random 2")
        # player4 = RandomPlayer(DurakEnv.HAND_SIZE, "random 3")
        game = DurakEnv([player1, player2, player3, player4], False)
        game_num = 0
        lost = 0
        tie = 0
        for game_index in range(num_games):
            state = game.reset()
            game.render()
            while True:
                turn_player = game.get_turn_player()
                to_attack = game.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done, info = game.step(act)
                state = new_state
                game.render()
                if done:
                    if game.get_loser():
                        print("loser is: " + game.get_loser().name)
                        if game.get_loser().name == player1.name:
                            lost += 1
                    break
        print("Lost: " + str(lost) + " out of: " + str(num_games))
    logging.shutdown()
