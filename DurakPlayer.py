from Deck import Deck


class DurakPlayer:

    def __init__(self, hand_size, name):
        self._hand = []
        self._trump_rank = None
        self.__hand_size = hand_size
        self.__name = name

    def take_cards(self, cards):
        self._hand = self._hand + cards

    def set_trump_rank(self, rank):
        self._trump_rank = rank

    def attack(self, state):
        raise NotImplementedError()

    def defend(self, state):
        raise NotImplementedError()

    def is_starting_hand_legal(self):
        num_hearts = 0
        num_diamonds = 0
        num_spades = 0
        num_clubs = 0
        for _, series in self._hand:
            if series == Deck.HEARTS:
                num_hearts += 1
            elif series == Deck.DIAMONDS:
                num_diamonds += 1
            elif series == Deck.SPADES:
                num_spades += 1
            else:
                num_clubs += 1
        if ((self.__hand_size - 1) in [num_hearts, num_diamonds, num_spades, num_clubs]) or \
                ((num_hearts + num_diamonds) == self.__hand_size) or \
                ((num_spades + num_clubs) == self.__hand_size):
            return False
        return True

    def empty_hand(self):
        self._hand = []

    def get_lowest_trump(self):
        min_trump = Deck.NO_CARD
        for value, series in self._hand:
            if series == self._trump_rank:
                if min_trump == Deck.NO_CARD or value < min_trump:
                    min_trump = value
        return min_trump

    @property
    def hand_size(self):
        return len(self._hand)

    @property
    def hand(self):
        return self._hand

    @property
    def name(self):
        return self.__name

    def __str__(self):
        return str(self._hand)


class BasePlayer(DurakPlayer):

    def attack(self, table):
        attacking, defending = table
        if len(attacking) == 0:
            attacking_card = self.__get_lowest_card(self._hand)
        else:
            legal_cards = list()
            for card in self._hand:
                if self.__is_value_on_table(card[0], table):
                    legal_cards.append(card)
            if len(legal_cards):
                attacking_card = self.__get_lowest_card(legal_cards)
            else:
                attacking_card = Deck.NO_CARD
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table):
        attacking, defending = table
        defend_from = attacking[-1]
        legal_cards = list()
        for card in self._hand:
            if (card[1] == defend_from[1]) and (defend_from[0] < card[0]):
                legal_cards.append(card)
            elif (card[1] == self._trump_rank) and (defend_from[1] != self._trump_rank):
                legal_cards.append(card)
        if len(legal_cards) == 0:
            defending_card = Deck.NO_CARD
        else:
            defending_card = self.__get_lowest_card(legal_cards)
            self._hand.remove(defending_card)
        return defending_card

    def __get_lowest_card(self, cards):
        lowest_card = cards[0]
        for card in cards[1:]:
            value, rank = card
            if (self._trump_rank not in [rank, lowest_card[1]]) or ((rank == self._trump_rank) and (lowest_card[1] == self._trump_rank)):
                if value < lowest_card[0]:
                    lowest_card = card
            elif lowest_card[1] == self._trump_rank:
                lowest_card = card
        return lowest_card

    @staticmethod
    def __is_value_on_table(value, table):
        for group in table:
            for card_value, _ in group:
                if card_value == value:
                    return True
        return False


class HumanPlayer(DurakPlayer):

    def attack(self, state):
        pass

    def defend(self, state):
        pass
