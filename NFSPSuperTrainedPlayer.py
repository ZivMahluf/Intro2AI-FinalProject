from NFSPPlayer import NFSPPlayer
from NFSPTrainedPlayer import TrainedNFSPPlayer
import copy


class NFSPSuperTrainedPlayer(TrainedNFSPPlayer):
    def __init__(self, hand_size, name, device='cpu'):
        super().__init__(hand_size,name)
        self.attackPlayer = TrainedNFSPPlayer(hand_size, name)
        self.defendPlayer = TrainedNFSPPlayer(hand_size, name)
        self.attacked = False

    def attack(self, table, legal_cards_to_play):
        self.attacked = True
        self.attackPlayer._hand = self._hand
        act =  self.attackPlayer.attack(table, legal_cards_to_play)
        self._hand = self.attackPlayer._hand
        return act

    def defend(self, table, legal_cards_to_play):
        self.attacked = False
        self.defendPlayer._hand = self._hand
        act =  self.defendPlayer.defend(table, legal_cards_to_play)
        self._hand = self.defendPlayer._hand
        return act

    def load_from_other_player(self, other):
        self.attackPlayer.policy = copy.deepcopy(other.attackPlayer.policy)
        self.defendPlayer.policy = copy.deepcopy(other.defendPlayer.policy)