from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional


class LearningPlayer(DurakPlayer):

    def learn_step(self, old_state, new_state, action, reward, info):
        raise NotImplementedError()

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        raise NotImplementedError()

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        raise NotImplementedError()
