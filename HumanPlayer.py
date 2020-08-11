from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional
import pygame
import random


class HumanPlayer(DurakPlayer):

    """
    A human player attacks and defends through a GUI with which they interact. If not GUI is defined for some reason, the player will act randomly.
    """

    def __init__(self, hand_size: int, name: str):
        """
        Constructor.
        :param hand_size: Number of cards in the initial hand of the player.
        :param name: Name of the player.
        """
        DurakPlayer.__init__(self, hand_size, name)
        self.__game_gui = None
        self.__set_gui = False

    def set_gui(self, gui) -> None:
        """
        Sets the gui with which the human player interacts.
        :param gui: The gui for the human player to interact with.
        """
        self.__game_gui = gui
        self.__set_gui = True

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        if self.__set_gui:
            return self.__get_card(legal_cards_to_play, "- Attack -")
        attacking_card = random.choice(legal_cards_to_play)
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        if self.__set_gui:
            return self.__get_card(legal_cards_to_play, "- Defend -")
        defending_card = random.choice(legal_cards_to_play)
        if defending_card != Deck.NO_CARD:
            self._hand.remove(defending_card)
        return defending_card

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

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.__game_gui.get_pass_button().collidepoint(event.pos) and selected_card in legal_cards_to_play:
                        return selected_card
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

    def __get_clicked_card(self) -> Optional[Deck.CardType]:
        """
        Detects which card the player clicked on and returns is.
        :return: The card in the player's hand which was clicked on.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        card_w, card_h = self.__game_gui.card_size
        positions = self.__game_gui.human_player_cards_positions
        pressed_card = Deck.NO_CARD
        if len(positions):
            if (positions[0][0] <= mouse_x < (positions[-1][0] + card_w)) and (positions[0][1] <= mouse_y <= (positions[0][1] + card_h)):
                for i, (x, _) in enumerate(positions):
                    if i < (len(positions) - 1):
                        if x <= mouse_x < min(x + card_w, positions[i + 1][0]):
                            pressed_card = self._hand[i]
                    else:
                        if x <= mouse_x <= (x + card_w):
                            pressed_card = self._hand[i]
        return pressed_card
