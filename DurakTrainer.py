from DurakGameRunner import DurakRunner
from DurakPlayer import DurakPlayer
from LearningPlayer import LearningPlayer
from HumanPlayer import HumanPlayer
from RandomPlayer import RandomPlayer
from BasicPlayer import BasicPlayer


class DurakTrainer:
    def __init__(self):
        self.learning_agents = list()
        self.other_agents = list()
        self.game_runner = DurakRunner()
        self.training_data = list()  # list of tuples ((prev_state, action, reward, next state), cards in deck, cards in players' hands (as a list of tuples of (name, list of cards)))

    def add_auto_agent(self, playerClass, name: str):
        player = playerClass(self.game_runner.HAND_SIZE, name)
        if (len(self.learning_agents) + len(self.other_agents)) < DurakRunner.MAX_PLAYERS and type(player) != HumanPlayer:
            if type(player) == LearningPlayer:
                self.learning_agents.append(player)
            else:
                self.other_agents.append(player)
            self.game_runner.add_player(player)

    def train_agents(self, episodes: int = 1, games_per_episode: int = 1, render: bool = True, verbose: bool = False):
        for episode in range(1, episodes + 1):
            if verbose:
                print("-------------------- Episode", episode, '--------------------')
                print("----------------- Running Games -----------------")
            self.game_runner.play_games(games_per_episode, render, verbose)
            if verbose:
                print("-------------- Analyzing Game Logs --------------")
            self.construct_learning_data()
            self.do_training()

    def construct_learning_data(self):
        """
        Uses the logs of the games to construct learning data for the learning agents.
        """
        games_log = self.game_runner.get_games_log()
        for game_log in games_log:
            for round_log in game_log:
                for record in round_log:
                    # The following information can reconstruct a full game state (including a memory of an all-knowing player regarding the cards in the game),
                    # and calculate an accurate reward for each action.
                    prev_state, prev_action, acting_player_name, next_state, attacker_name, defender_name, cards_in_deck, player_hands = record
                    print("successfully got information from record")
                    break
                break
            break

    def calculate_reward(self) -> float:
        """
        A reward function which gives a reward to a player for an action based on the given information.
        :return: A reward for an action at a given state.
        """
        pass

    def do_training(self):
        """
        Calls the learning methods of the learning players.
        """
        pass
