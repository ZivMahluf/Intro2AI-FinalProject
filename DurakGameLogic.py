from Deck import Deck
from typing import Tuple, List


class DurakLogic:
    """
    This class holds function which enforce the logic of a Durak game.
    """

    @staticmethod
    def get_legal_attacking_cards(attacker_hand: List[Deck.CardType], table: Tuple[List[Deck.CardType], List[Deck.CardType]]) -> List[Deck.CardType]:
        """
        :param attacker_hand: The cards in the hand of the attacker.
        :param table: The cards on the table, as a tuple (attacking, defending)
        :return: List of cards from the given hand that are legal to attack with (if the table is empty - any card in the hand, otherwise - any card whose value is on the table, or no card at all).
        """
        if len(table[0]) == 0:
            return attacker_hand[:]
        legal_attacking_cards = [Deck.NO_CARD]
        for card in attacker_hand:
            for i in range(len(table)):
                for card_on_table in table[i]:
                    if card[0] == card_on_table[0] and card not in legal_attacking_cards:
                        legal_attacking_cards.append(card)
                        break
        return legal_attacking_cards

    @staticmethod
    def get_legal_defending_cards(defender_hand: List[Deck.CardType], attacking_card: Deck.CardType, trump_suit: int) -> List[Deck.CardType]:
        """
        :param defender_hand: The cards in the defending player's hand.
        :param attacking_card: The card from which to defend.
        :param trump_suit: The trump suit in the game.
        :return: List of cards from the defender's hand which can be used to defend from the attacking card, including no card.
        """
        legal_defending_cards = [Deck.NO_CARD]
        for card in defender_hand:
            if attacking_card[1] == card[1]:
                if attacking_card[0] < card[0]:
                    legal_defending_cards.append(card)
            elif (attacking_card[1] != trump_suit) and (card[1] == trump_suit):
                legal_defending_cards.append(card)
        return legal_defending_cards
