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
            chose_card = legal_cards_to_play[0]
        else:
            choose_from = legal_cards_to_play[:]
            if Deck.NO_CARD in choose_from:
                choose_from.remove(Deck.NO_CARD)
            chose_card = choose_from[0]
            for card in choose_from[1:]:
                value, suit = card
                if (self._trump_suit not in [suit, chose_card[1]]) or ((suit == self._trump_suit) and (chose_card[1] == self._trump_suit)):
                    if value < chose_card[0]:
                        chose_card = card
                elif chose_card[1] == self._trump_suit:
                    chose_card = card
        if chose_card != Deck.NO_CARD:
            self._hand.remove(chose_card)
        return chose_card
