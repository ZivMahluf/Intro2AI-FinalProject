from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional


class BasicPlayer(DurakPlayer):
    """
    A player of this class always plays the weakest card possible from the given list of legal cards.
    """

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.__do_basic_play(legal_cards_to_play)

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.__do_basic_play(legal_cards_to_play)

    def __do_basic_play(self, legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        """
        Chooses the weakest card from the given list of cards.
        The weakest card is the card with the lowest value. Any trump card has a higher value than any non-trump card.
        :param legal_cards_to_play: A list of available cards to choose from (might include Deck.NO_CARD).
        :return: The card with the lowest value from the legal cards.
        """
        if len(legal_cards_to_play) == 1:
            return Deck.NO_CARD
        lowest_card = legal_cards_to_play[1]
        for card in legal_cards_to_play[2:]:
            value, rank = card
            if (self._trump_rank not in [rank, lowest_card[1]]) or ((rank == self._trump_rank) and (lowest_card[1] == self._trump_rank)):
                if value < lowest_card[0]:
                    lowest_card = card
            elif lowest_card[1] == self._trump_rank:
                lowest_card = card
        self._hand.remove(lowest_card)
        return lowest_card
