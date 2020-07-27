from DurakGameRunner import DurakRunner
from DurakPlayer import DurakPlayer
from LearningPlayer import LearningPlayer
from HumanPlayer import HumanPlayer
from RandomPlayer import RandomPlayer
from BasicPlayer import BasicPlayer
from SanityPlayer import SanityCheckPlayer
from Deck import Deck
from typing import Union


class DurakTrainer:
    def __init__(self):
        self.learning_agents = list()
        self.other_agents = list()
        self.game_runner = DurakRunner()
        self.training_data_per_player = dict()
        self.update_args_by_game = list()
        self.verbose = False

    def add_auto_agent(self, playerClass, name: str):
        player = playerClass(self.game_runner.HAND_SIZE, name)
        if (len(self.learning_agents) + len(self.other_agents)) < DurakRunner.MAX_PLAYERS and type(player) != HumanPlayer:
            if isinstance(player, LearningPlayer):
                self.learning_agents.append(player)
                self.training_data_per_player[name] = list()
            else:
                self.other_agents.append(player)
            self.game_runner.add_player(player)

    def train_agents(self, episodes: int = 1, games_per_episode: int = 1, render: bool = False, verbose: bool = False):
        self.verbose = verbose
        for episode in range(1, episodes + 1):
            self.update_args_by_game = list()
            for player_name in self.training_data_per_player:
                self.training_data_per_player[player_name] = list()
            if verbose:
                print("-------------------- Episode", episode, '--------------------')
                print("----------------- Running Games -----------------")
            self.game_runner.play_games(games_per_episode, render, False)
            if verbose:
                print("-------------- Analyzing Game Logs --------------")
            self.construct_learning_data()
            self.do_training()

    def construct_learning_data(self):
        """
        Uses the logs of the games to construct learning data for the learning agents.
        """
        games_log = self.game_runner.get_games_log()
        for g, game_log in enumerate(games_log):
            if self.verbose:
                print('analyzing log of game', g + 1)
            update_args = list()
            for round_log in game_log:
                for i, record in enumerate(round_log):
                    # The following information can reconstruct a full game state (including a memory of an all-knowing player regarding the cards in the game),
                    # and calculate an accurate reward for each action.
                    prev_state, prev_action, acting_player_name_hand, next_state, attacker_name, defender_name, cards_in_deck, player_hands, trump_rank = record
                    acting_player_name, acting_player_hand = acting_player_name_hand
                    update_args.append((acting_player_name, prev_action, prev_action == next_state[0][-1]))
                    if i == (len(round_log) - 1):
                        update_args.append((defender_name, next_state, len(next_state[0]) == len(next_state[1])))
                    if acting_player_name in self.training_data_per_player:
                        next_next_state = (list(), list()) if i == (len(round_log) - 1) else round_log[i + 1]
                        reward = self.calculate_reward(len(round_log), i, prev_action, len(acting_player_hand), next_state, len(cards_in_deck), trump_rank, next_next_state)
                        self.training_data_per_player[acting_player_name].append((prev_state, prev_action, reward, next_state))
            self.update_args_by_game.append(update_args[:])

    @staticmethod
    def calculate_reward(round_log_length, index, prev_action, acting_player_hand_size, next_state, deck_size, trump_rank, next_next_state) -> Union[int, float]:
        """
        Calculates a reward for the acting player based on the given parameters.
        :param round_log_length: Number of round actions in the round.
        :param index: Index of the current action, starting from 0.
        :param prev_action: The last action taken.
        :param acting_player_hand_size: Number of cards in the acting player's hand.
        :param next_state: The next state.
        :param deck_size: Number of remaining cards in the deck.
        :param trump_rank: Trump rank of the game.
        :param next_next_state: The next_state field from the next entry in the log (only accessed if a next entry exists).
        :return: The calculated reward for the action.
        """
        if acting_player_hand_size == 0 and deck_size == 0:
            reward = 50  # arbitrary high reward for winning
        elif index == 0:
            reward = -prev_action[0]
            if prev_action[1] == trump_rank:
                reward -= Deck.ACE
        elif index == (round_log_length - 1):
            if len(next_state[0]) == len(next_state[1]):
                # the defending player succeeded and is the last player to act
                reward = 20
            else:
                # defence unsuccessful (the defender is not the last player to act)
                reward = prev_action[0]
                if prev_action[1] == trump_rank and deck_size > 0:
                    reward *= 0.9  # penalty for giving a trump card to an opponent when there are still cards in the deck
                if prev_action[1] == trump_rank and acting_player_hand_size > 0:
                    reward *= 0.85  # penalty for giving a trump card to an opponent that doesn't empty the hand (meaning it might be used against the player)
        else:
            if prev_action in next_state[0]:
                # the last player attacked
                if len(next_next_state[0]) > len(next_state[0]):
                    # two or more attacks were made in a row - defence failed
                    reward = prev_action[0]
                    if prev_action[1] == trump_rank and deck_size > 0:
                        reward *= 0.9  # penalty for giving a trump card to an opponent when there are still cards in the deck
                    if prev_action[1] == trump_rank and acting_player_hand_size > 0:
                        reward *= 0.85  # penalty for giving a trump card to an opponent that doesn't empty the hand (meaning it might be used against the player)
                else:
                    # the defender defended against the previous attacking card
                    reward = -prev_action[0]
                    if prev_action[1] == trump_rank and deck_size > 0:
                        reward -= (prev_action[0] / 2)  # penalty for attacking with a trump card when the deck is not empty
                    if prev_action[1] == trump_rank and acting_player_hand_size > 0:
                        reward -= (prev_action[0] / 2)  # penalty for attacking with a trump card when hand is not empty
            else:
                # the last player defended
                reward = prev_action[0]
                if prev_action[1] == trump_rank and next_state[0][-1][1] != trump_rank:
                    reward *= 0.9  # penalty for defending with a trump card against a non-trump card
        return reward

    def do_training(self):
        """
        Trains the learning agents using the progress of the games as follows:
        For each player, a replay of all games in the episode occurs by resetting the player for each game, and updating its memory by repeatedly calling the player's
        update methods (update_round_progress, and update_end_round). During the replay o each game, when reaching an action done by the player for which the replay is played
        (which is recognized by arg1 - which is the acting player's name - being the name of the learning player)
        """
        if self.verbose:
            print('-------------- Training players --------------')
        for player in self.learning_agents:
            if self.verbose:
                print('training', player.name)
            training_data = self.training_data_per_player[player.name]
            index = 0
            end = False
            for game_update_args in self.update_args_by_game:
                player.initialize_for_game()
                for arg1, arg2, arg3 in game_update_args:
                    if type(arg2[0]) == list:
                        # in this case, arg2 represents the table - a tuple of two lists.
                        player.update_end_round(arg1, arg2, arg3)
                    else:
                        # in this case, arg2 represents a card - a tuple of ints.
                        player.update_round_progress(arg1, arg2, arg3)
                        (prev_state, prev_action, reward, next_state) = training_data[index]
                        if player.name == arg1:
                            # in this case, the last action which took place was done by the learning player, meaning that the replay caught up to the current action from which the player
                            # will learn (in training_data[index]), and the players internal state should be the same as it was when the action first took place during the game, meaning that
                            # at this point, we can call the learning method so that the agent learns from the action under the same circumstances that were in the real game.
                            player.learn(prev_state, prev_action, reward, next_state)
                            index += 1
                        if index == len(training_data):
                            end = True
                            break
                if end:
                    break


trainer = DurakTrainer()
trainer.add_auto_agent(SanityCheckPlayer, 'SanityCheck')
trainer.add_auto_agent(BasicPlayer, 'BasicBitch')
trainer.train_agents(5000, 10, verbose=True)
