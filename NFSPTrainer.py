from NFSPPlayer import NFSPPlayer
from DurakEnv import DurakEnv
import os


hand_size = DurakEnv.HAND_SIZE


class NFSPTrainer:
    def __init__(self):
        self.learning_players = [NFSPPlayer(hand_size, "NFSP-1"), NFSPPlayer(hand_size, "NFSP-2"), NFSPPlayer(hand_size, "NFSP-3")]
        self.file_paths = [os.path.join('SavedModels', 'NFSP-1'), os.path.join('SavedModels', 'NFSP-2'), os.path.join('SavedModels', 'NFSP-3')]
        self.env = DurakEnv(self.learning_players)
        self.test_losers = list()

    def get_learning_players_names(self):
        return [player.name for player in self.learning_players]

    def get_learning_players(self):
        return self.learning_players

    def train(self, games):
        for i in range(games):
            state = self.env.reset()
            self.env.render()
            while True:
                turn_player = self.env.get_turn_player()
                to_attack = self.env.to_attack()
                act = turn_player.get_action(state, to_attack)
                new_state, reward, done, info = self.env.step(act)
                if turn_player in self.learning_players:
                    turn_player.learn_step(state, new_state, act, reward, info)
                state = new_state
                self.env.render()
                if done:
                    break

    def test(self, games):
        losses = {player.name: 0 for player in self.learning_players}  # for loss ratios
        for i in range(games):
            print('test game', i + 1)
            state = self.env.reset()
            self.env.render()
            while True:
                turn_player = self.env.get_turn_player()
                to_attack = self.env.to_attack()
                act = turn_player.get_action(state, to_attack)
                state, _, done, _ = self.env.step(act)
                self.env.render()
                if done:
                    break
            loser = self.env.get_loser()
            if loser is not None and loser.name in losses:
                losses[loser.name] += 1
        for name in losses:
            losses[name] /= games
        return losses
