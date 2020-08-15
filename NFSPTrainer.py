from NFSPPlayer import NFSPPlayer
from DurakEnv import DurakEnv
from DurakPlayer import DurakPlayer
from Types import List


class NFSPTrainer:
    def __init__(self, learning_players: List[NFSPPlayer], other_players: List[DurakPlayer]):
        """
        Constructor.
        :param learning_players: a list of players which will be learning during the training.
        :param other_players: a list of additional players for the learning players to train with.
        """
        # only taking up to DurakEnv.MAX_PLAYERS players, starting from the learning players
        self.learning_players = learning_players[:DurakEnv.MAX_PLAYERS]
        self.other_players = other_players[:(
            DurakEnv.MAX_PLAYERS - len(self.learning_players))]
        self.env = DurakEnv(self.other_players + self.learning_players, False)
        self.test_losers = list()

    def train(self, games) -> None:
        """
        Trains the learning players for the given number of games.
        :param games: Number of training games.
        """
        for i in range(games):
            state = self.env.reset()
            count = 0
            while True:
                count += 1
                turn_player = self.env.get_turn_player()
                to_attack = self.env.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done = self.env.step(act)
                if isinstance(turn_player, NFSPPlayer):
                    turn_player.learn_step(state, new_state, act, reward, self.env.turn_player.last_hand)
                state = new_state
                if done or count > 300:
                    break
            for player in self.env.players:
                if isinstance(player, NFSPPlayer):
                    player.end_game()
