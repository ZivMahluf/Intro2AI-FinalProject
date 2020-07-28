import pygame
import os
from math import sin, cos, radians
from DurakPlayer import DurakPlayer
from HumanPlayer import HumanPlayer
from Deck import Deck
from typing import List, Union, Tuple


class GUI:

    """
    Parameters regarding the size of the game screen.
    """
    __side_length = 750
    __human_player_cards_menu_height = 120
    """
    Parameters regarding spacing within the game screen.
    """
    __radius = __side_length // 2.3
    __size = 30
    __horizontal_space = 10
    __vertical_space = 25

    def __init__(self):
        """
        Constructor.
        """
        pygame.init()
        self.__screen = pygame.display.set_mode((self.__side_length, self.__side_length + self.__human_player_cards_menu_height))
        self.__init_images()
        self.__init_positions()
        self.__font = pygame.font.Font(pygame.font.get_default_font(), self.__size)
        self.__human_player_cards_positions = list()
        self.__pass_button = pygame.Rect(550, 680, 150, 50)
        self.__pass_button_text = self.__font.render("Pass", True, (0, 0, 0))
        self.__pass_button_rect = self.__pass_button_text.get_rect()
        self.__pass_button_rect.center = (625, 705)

    def __init_images(self) -> None:
        """
        Loads all images required for the GUI of the game.
        """
        deck = Deck.get_full_list_of_cards()
        self.__card_images = dict()
        self.__card_size = None
        for card in deck:
            if card[0] < Deck.JACK:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", str(card[0]) + Deck.STRING_RANKS[card[1]] + ".png"))
            elif card[0] == Deck.JACK:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "J" + Deck.STRING_RANKS[card[1]] + ".png"))
            elif card[0] == Deck.QUEEN:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "Q" + Deck.STRING_RANKS[card[1]] + ".png"))
            elif card[0] == Deck.KING:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "K" + Deck.STRING_RANKS[card[1]] + ".png"))
            else:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "A" + Deck.STRING_RANKS[card[1]] + ".png"))
            if self.__card_size is None:
                self.__card_size = self.__card_images[card].get_size()
        self.__rank_images = dict()
        for rank in Deck.RANKS:
            if rank == Deck.HEARTS:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Hearts.png"))
            elif rank == Deck.CLUBS:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Clubs.png"))
            elif rank == Deck.DIAMONDS:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Diamonds.png"))
            elif rank == Deck.SPADES:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Spades.png"))
        self.__table_image = pygame.image.load(os.path.join("Images", "Table.png"))
        self.__back_of_card_image = pygame.image.load(os.path.join("Images", "ZCardBack.png"))

    def __init_positions(self) -> None:
        """
        Initializes the positions of the names of players, the cards on the table, and the deck placement on the table.
        """
        deck = Deck.get_full_list_of_cards()
        self.__num_players_to_positions = {i: [(self.__side_length // 2 + int(self.__radius * cos(radians(90 + j * (360 // i)))),
                                                self.__side_length // 2 - 35 + int(self.__radius * sin(radians(90 + j * (360 // i)))))
                                               for j in range(i)] for i in range(len(deck) + 1)}
        self.__initial_card_position = (self.__side_length // 2 - 3 * (self.__card_size[0] + self.__horizontal_space), self.__side_length // 2 - (3 * self.__card_size[1]) // 2)
        self.__deck_initial_pos = (self.__side_length // 2 - self.__card_size[0] // 2, self.__side_length // 2 + self.__card_size[1] // 2)

    def show_screen(self, players: List[DurakPlayer], table: Tuple[Union[list, List[Deck.CardType]], Union[list, List[Deck.CardType]]],
                    attacker: Union[DurakPlayer, None], defender: Union[DurakPlayer, None], deck: Deck, trump_rank: int) -> None:
        """
        Shows the current state of the game.
        :param players: A list of the players in the game.
        :param table: A tuple of two lists of cards on the table - attacking cards, and defending cards.
        :param attacker: The attacking player.
        :param defender: The defending player.
        :param deck: The deck object used by the game.
        :param trump_rank: The trump rank in the game.
        """
        self.__screen.fill((0, 0, 0))
        self.__screen.blit(self.__table_image, (0, 0))
        self.__screen.blit(self.__rank_images[trump_rank], (0, 0))
        self.__human_player_cards_positions = list()

        # Show pass button
        pygame.draw.rect(self.__screen, [255, 255, 255], self.__pass_button)
        self.__screen.blit(self.__pass_button_text, self.__pass_button_rect)

        # Show players
        for i, position in enumerate(self.__num_players_to_positions[len(players)]):
            # Show names
            if players[i] == attacker:
                text = self.__font.render(players[i].name, True, (255, 0, 0))
            elif players[i] == defender:
                text = self.__font.render(players[i].name, True, (0, 0, 255))
            else:
                text = self.__font.render(players[i].name, True, (0, 0, 0))
            rect = text.get_rect()
            rect.center = position
            self.__screen.blit(text, rect.topleft)
            # Depending on the type of the player, shows the hand of the player (for non-human players, shows the back of the cards)
            if type(players[i]) == HumanPlayer:
                # Cards of the human player are displayed on the bottom menu of the screen.
                human_player_horizontal_space = 0
                if players[i].hand_size:
                    human_player_horizontal_space = (self.__side_length - 20) // players[i].hand_size
                for j, card in enumerate(players[i].hand):
                    self.__human_player_cards_positions.append((10 + j * human_player_horizontal_space, self.__side_length + 15))
                    self.__screen.blit(self.__card_images[card], self.__human_player_cards_positions[-1])
            else:
                # For automatic agents, the backs of the cards in their hands are displayed under their name.
                for j in range(players[i].hand_size):
                    self.__screen.blit(self.__back_of_card_image, (rect.left + j * 4, rect.bottom))
        # Shows the cards on the table
        for i in range(len(table)):
            for j in range(len(table[i])):
                self.__screen.blit(self.__card_images[table[i][j]], (self.__initial_card_position[0] + j * (self.__horizontal_space + self.__card_size[0]),
                                                                     self.__initial_card_position[1] + i * self.__vertical_space))
        # Shows the deck (face down, on the table)
        for i in range(deck.current_num_cards):
            self.__screen.blit(self.__back_of_card_image, (self.__deck_initial_pos[0], self.__deck_initial_pos[1] - i))
        # Updates the screen, and waits (the waiting is for people to be able to follow the game)
        pygame.display.flip()
        pygame.time.wait(1000)

    def get_pass_button(self) -> pygame.Rect:
        """
        Returns: The pass button pygame.Rect object
        """
        return self.__pass_button

    def show_message(self, message: str) -> None:
        """
        Displays a message under the name of the first player (which is located at the bottom of the table).
        This position was chosen since this method is only used for displaying messages to the human player in the place where normally the cards of an
        automatic agent would be rendered.
        :param message: A message to display.
        """
        text = self.__font.render(message, True, (150, 150, 150))
        rect = text.get_rect()
        human_player_text_position = self.__num_players_to_positions[1][0]
        rect.center = (human_player_text_position[0], human_player_text_position[1] + self.__size + 10)
        self.__screen.blit(text, rect.topleft)
        pygame.display.flip()

    @staticmethod
    def dispose_events():
        pygame.event.get()

    @staticmethod
    def pressed_quit() -> bool:
        """
        :return: Weather the user pressed the 'x' button on the pygame window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    @staticmethod
    def end() -> None:
        """
        Ends the pygame session.
        """
        pygame.quit()

    @property
    def card_size(self) -> Tuple[int, int]:
        """
        :return: The size of a card image - (width, height), in pixels.
        """
        return self.__card_size

    @property
    def human_player_cards_positions(self) -> List[Tuple[int, int]]:
        """
        :return: A list containing the (x, y) coordinates of the top left corner of each card rendered in the human player's hand the last time the method to show the screen was called.
        """
        return self.__human_player_cards_positions
