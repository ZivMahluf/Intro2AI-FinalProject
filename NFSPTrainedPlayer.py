from NFSPModel import DQNBase, Policy
from Deck import Deck
import torch
from NFSPPlayer import NFSPPlayer
from Types import TableType, CardListType, CardType
import copy


class TrainedNFSPPlayer(NFSPPlayer):

    def __init__(self, hand_size: int, name: str):
        """
        Constructor.
        :param hand_size: initial hand size.
        :param name: player name.
        """
        super().__init__(hand_size, name)
        self.policy = Policy()
        self.current_model = DQNBase()

    def load_from_other_player(self, other: NFSPPlayer) -> None:
        """
        Loads policy from another player.
        :param other: player to load policy from.
        """
        self.policy = copy.deepcopy(other.policy)

    def learn_step(self, old_state, new_state, action, reward, old_hand):
        """
        Does not preform learning step, since this is a trained player.
        """
        pass

    def end_game(self):
        """
        Does not update at the end of the game, since the player is trained.
        """
        pass

    def act(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        """
        Chooses an action to do based on the given table and the legal cards to play.
        :param table: (attacking cards, defending cards, number of cards in deck, number of cards in each player's hand)
        :param legal_cards_to_play: list of legal cards to play out of the player's hand, and might include Deck.NO_CARD.
        :return: The chosen card to play.
        """
        legal_cards_vec, state = NFSPPlayer.get_network_input(legal_cards_to_play, table, self.discard_pile, self._hand)
        card = Deck.get_card_from_index(self.policy.act(torch.FloatTensor(state), 0, legal_cards_vec))
        if card[0] != -1:
            self._hand.remove(card)
        return card

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        return self.act(table, legal_cards_to_play)

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        return self.act(table, legal_cards_to_play)
