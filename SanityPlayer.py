from typing import Tuple, List, Union
from Deck import Deck
from LearningPlayer import LearningPlayer


class SanityCheckPlayer(LearningPlayer):

    def first_initialize(self, players_names, full_deck_size) -> None:
        pass

    def initialize_for_game(self) -> None:
        pass

    def learn(self, prev_table: Tuple[List[Deck.CardType], List[Deck.CardType]], prev_action: Deck.CardType, reward: float, next_table: Tuple[List[Deck.CardType], List[Deck.CardType]]) -> None:
        pass

    def batch_learn(self, batch: List[Tuple[Tuple[List[Deck.CardType], List[Deck.CardType]], Deck.CardType, Union[int, float], Tuple[List[Deck.CardType], List[Deck.CardType]]]]):
        for prev_state, prev_action, reward, next_state in batch:
            self.learn(prev_state, prev_action, reward, next_state)
