from NFSPModel import DQN, Policy
import torch
from NFSPPlayer import NFSPPlayer
from Types import TableType, CardListType, CardType
import copy


class TrainedNFSPPlayer(NFSPPlayer):

    def __init__(self, hand_size, name):
        super().__init__(hand_size, name)
        self.policy = Policy()
        self.current_model = DQN()

    def load_from_other_player(self, other: NFSPPlayer) -> None:
        self.policy = copy.deepcopy(other.policy)

    def learn_step(self, old_state, new_state, action, reward, old_hand):
        pass

    def end_game(self):
        pass

    def act(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        legal_cards_vec, state = NFSPPlayer.get_network_input(legal_cards_to_play, table, self.discard_pile, self._hand)
        card = NFSPPlayer.action_to_card(self.policy.act(torch.FloatTensor(state), 0, legal_cards_vec))
        if card[0] != -1:
            self._hand.remove(card)
        return card

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        return self.act(table, legal_cards_to_play)

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        return self.act(table, legal_cards_to_play)

    def initialize_for_game(self) -> None:
        super().initialize_for_game()
