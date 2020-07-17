from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional, np
from typing import Dict, Union
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import random


class LearningPlayer(DurakPlayer):
    def __init__(self, hand_size: int, name: str, num_actions: int, cards_indices: Dict[Union[Deck.CardType, type(Deck.NO_CARD)], int]):
        DurakPlayer.__init__(self, hand_size, name)
        self.initialized = False
        self.num_actions = num_actions
        self.cards_memory = {}
        self.cards_indices = cards_indices
        self.state_memory = deque(maxlen=500)
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.num_actions, activation='linear'))
        self.input_shape = None

    def first_initialize(self, players_names: List[str]):
        self.cards_memory['unknown'] = list()
        self.cards_memory['discarded'] = list()
        for name in players_names:
            self.cards_memory[name] = list()
        self.input_shape = (len(self.cards_memory) + 2, len(Deck.get_full_list_of_cards()))
        self.initialized = True

    def initialize_for_game(self):
        for key in self.cards_memory.keys():
            self.cards_memory[key] = list()
        self.cards_memory['unknown'] = Deck.get_full_list_of_cards()
        for card in self.hand:
            self.cards_memory['unknown'].remove(card)
            self.cards_memory[self.name].append(card)

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return random.choice(legal_cards_to_play)

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return random.choice(legal_cards_to_play)

    def to_features(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]]) -> np.ndarray:
        pass

    def update_round_progress(self, player_name: str, played_card: Deck.CardType) -> None:
        """
        Updates the agent about a card that was played by a player.
        :param player_name: Name of the player that played.
        :param played_card: The card played by that player.
        """
        pass

    def update_end_round(self, defending_player_name: str, table: Tuple[List[Deck.CardType], List[Deck.CardType]], successfully_defended: bool) -> None:
        """
        Updates the agent about the result of the round - weather the defending player defended successfully or not.
        :param defending_player_name: Defending player's name
        :param table: Cards on the table at the end of the round (before clearing)
        :param successfully_defended: Weather the defence was successful (which means all cards are discarded), or not (which means the defending player took all cards on the table).
        """
        pass

    def learn(self, prev_table, prev_action, reward, next_table):
        pass
