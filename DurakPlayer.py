from Deck import Deck
from typing import List, Tuple, Set, Optional, Dict, NewType
import random
import pygame
# Optional[ObjectType] means it can be None or ObjectType

# todo: maybe define a new type for table instead of using "table: Tuple[List[Deck.CardType], List[Deck.CardType]]"
# TableType = NewType('TableType', Tuple[List[Deck.CardType], List[Deck.CardType]])


class DurakPlayer:

    def __init__(self, hand_size: int, name: str):
        """
        Constructor.
        :param hand_size: Initial hand size.
        :param name: Name of the player.
        """
        self._hand = []
        self._trump_rank = None
        self.__initial_hand_size = hand_size
        self.__name = name

    def take_cards(self, cards: List[Deck.CardType]) -> None:
        """
        Adds the cards to the hand.
        :param cards: A list of cards to add.
        """
        self._hand = self._hand + cards

    def set_trump_rank(self, rank: int) -> None:
        """
        Sets the trump rank as the given rank.
        :param rank: Rank to set as trump rank.
        """
        self._trump_rank = rank

    def attack(self, state, legal_cards_to_play) -> Deck.CardType:
        raise NotImplementedError()

    def defend(self, state, legal_cards_to_play) -> Deck.CardType:
        raise NotImplementedError()

    def is_starting_hand_legal(self) -> bool:
        """
        A legal starting hand is one in which there are at most (initial hand size - 2) cards of each rank, and there is at least one 'red'
        card (hearts or diamonds), and one 'black' card (spades or clubs).
        :return: Weather the cards in the hand form a legal starting hand.
        """
        num_hearts = 0
        num_diamonds = 0
        num_spades = 0
        num_clubs = 0
        for _, rank in self._hand:
            if rank == Deck.HEARTS:
                num_hearts += 1
            elif rank == Deck.DIAMONDS:
                num_diamonds += 1
            elif rank == Deck.SPADES:
                num_spades += 1
            else:
                num_clubs += 1
        if ((self.__initial_hand_size - 1) in [num_hearts, num_diamonds, num_spades, num_clubs]) or \
                ((num_hearts + num_diamonds) == self.__initial_hand_size) or \
                ((num_spades + num_clubs) == self.__initial_hand_size):
            return False
        return True

    def empty_hand(self) -> None:
        """
        Empties the current hand.
        """
        self._hand = []

    def get_lowest_trump(self) -> int:
        """
        :return: The value of the lowest card with a trump rank, or Deck.NO_CARD is no card has a trump rank in the hand.
        """
        min_trump = Deck.NO_CARD
        for value, series in self._hand:
            if series == self._trump_rank:
                if min_trump == Deck.NO_CARD or value < min_trump:
                    min_trump = value
        return min_trump

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

    @property
    def hand_size(self) -> int:
        """
        :return: Number of cards in the current hand.
        """
        return len(self._hand)

    @property
    def hand(self) -> List[Deck.CardType]:
        """
        :return: A list of all cards currently in the hand of the player.
        """
        return self._hand

    @property
    def name(self) -> str:
        """
        :return: The name of the player.
        """
        return self.__name

    def __str__(self) -> str:
        """
        :return: String representation of the player (as a string representation of the hand)
        """
        return str(self._hand)


class BasePlayer(DurakPlayer):

    """
    A player of this class always plays the weakest card possible from the given list of legal cards.
    """

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        return self.__do_basic_play(legal_cards_to_play)

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        return self.__do_basic_play(legal_cards_to_play)

    def __do_basic_play(self, legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
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


class DurakPlayerWithMemory(DurakPlayer):
    def __init__(self, hand_size: int, name: str, other_players_names: List[str]):
        """
        constructor
        hand_size: number of cards you start with
        name: player's name
        num_players: a list that contains the names of the other players
        """
        DurakPlayer.__init__(self, hand_size, name)
        self.other_players_hand = dict()
        for player in other_players_names:
            self.other_players_hand[player] = set()
        self.discard_pile = set()

    def add_cards_to(self, name: str, cards: Set[Deck.CardType]) -> None:
        """
        Adds [cards] to [name]'s hand.
        :param name: The player that gets the set of cards
        :param cards: A set of cards to remember
        """
        if name not in self.other_players_hand:
            raise Exception("This player doesn't exist")
        # The cards in [name]'s hand are the union of his previous hand with [cards]
        self.other_players_hand[name] = self.other_players_hand[name] | cards

    def remove_cards_from(self, name: str, cards: Set[Deck.CardType]) -> None:
        """
        Removes [cards] from [name]'s hand.
        :param name: The player that gets rid of the set of cards
        :param cards: A set of cards that goes to the discard pile
        """
        if name not in self.other_players_hand:
            raise Exception("This player doesn't exist")
        # The cards in [name]'s hand are the cards that were previously in his hand minus the new ones
        self.other_players_hand[name] = self.other_players_hand[name] - cards
        self.discard_pile = self.discard_pile | cards


class RandomPlayer(DurakPlayer):
    """
    Random agent
    """
    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        """
        chooses a random card from legal cards to play
        """
        attacking_card = random.choice(legal_cards_to_play)
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        """
        chooses a random card from legal cards to play
        """
        defending_card = random.choice(legal_cards_to_play)
        if defending_card != Deck.NO_CARD:
            self._hand.remove(defending_card)
        return defending_card


class HumanPlayer(DurakPlayer):

    def __init__(self, hand_size: int, name: str, gui):
        """
        Constructor.
        :param hand_size: Number of cards in the initial hand of the player.
        :param name: Name of the player.
        :param gui: GUI object of the game.
        """
        DurakPlayer.__init__(self, hand_size, name)
        self.__game_gui = gui

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        return self.__get_card(legal_cards_to_play, "- Attack -")

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        return self.__get_card(legal_cards_to_play, "- Defend -")

    def __get_card(self, legal_cards_to_play: List[Deck.CardType], message: str = "") -> Optional[Deck.CardType]:
        """
        Gets a card selected by the player, and removes it from the hand of the player.
        A left click on a legal card in the hand selects it, and a right click anywhere selects to play Deck.NO_CARD. If the selected card
        (including Deck.NO_CARD) is not in the list of legal cards, nothing happens.
        :param legal_cards_to_play: List of legal cards to choose from.
        :param message: A message to display to the player.
        :return: A card to play (can also return Deck.NO_CARD), or None in case the player pressed the x button to quit the game.
        """
        self.__game_gui.show_message(message)
        waiting = True
        selected_card = Deck.NO_CARD
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        pressed_card = self.__get_clicked_card()
                        if pressed_card != Deck.NO_CARD:
                            if pressed_card in legal_cards_to_play:
                                selected_card = pressed_card
                                waiting = False
                    elif event.button == 3:
                        if Deck.NO_CARD in legal_cards_to_play:
                            waiting = False
        if selected_card != Deck.NO_CARD:
            self._hand.remove(selected_card)
        return selected_card

    def __get_clicked_card(self) -> Deck.CardType:
        """
        Detects which card the player clicked on and returns is.
        :return: The card in the player's hand which was clicked on.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        card_w, card_h = self.__game_gui.card_size
        positions = self.__game_gui.human_player_cards_positions
        pressed_card = Deck.NO_CARD
        if (positions[0][0] <= mouse_x < (positions[-1][0] + card_w)) and (positions[0][1] <= mouse_y <= (positions[0][1] + card_h)):
            for i, (x, _) in enumerate(positions):
                if i < (len(positions) - 1):
                    if x <= mouse_x < min(x + card_w, positions[i + 1][0]):
                        pressed_card = self._hand[i]
                else:
                    if x <= mouse_x <= (x + card_w):
                        pressed_card = self._hand[i]
        return pressed_card
