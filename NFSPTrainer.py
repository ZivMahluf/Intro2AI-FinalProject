from NFSPPlayer import NFSPPlayer
from DurakEnv import DurakEnv
from DurakPlayer import DurakPlayer
from Types import List


class NFSPTrainer:
    def __init__(self, learning_players: List[NFSPPlayer], other_players: List[DurakPlayer]):
        # only taking up to DurakEnv.MAX_PLAYERS players, starting from the learning players
        self.learning_players = learning_players[:DurakEnv.MAX_PLAYERS]
        self.other_players = other_players[:(DurakEnv.MAX_PLAYERS - len(self.learning_players))]
        self.env = DurakEnv(self.other_players + self.learning_players)
        self.test_losers = list()

    def get_learning_players_names(self) -> List[str]:
        return [player.name for player in self.learning_players]

    def get_learning_players(self) -> List[NFSPPlayer]:
        return self.learning_players

    def get_all_players(self) -> List[DurakPlayer]:
        return self.other_players + self.learning_players

    def train(self, games) -> None:
        for i in range(games):
            state = self.env.reset()
            self.env.render()
            count = 0
            while True:
                count += 1
                turn_player = self.env.get_turn_player()
                to_attack = self.env.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done = self.env.step(act)
                if isinstance(turn_player, NFSPPlayer):
                    turn_player.learn_step(state, new_state, act, reward)
                state = new_state
                self.env.render()
                if done or count > 300:
                    break
