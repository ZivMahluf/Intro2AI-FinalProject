import pygame
import os
from math import sin, cos, radians
from DurakPlayer import HumanPlayer


class GUI:

    __side_length = 750
    __human_player_cards_menu_height = 120
    __radius = __side_length // 2.3
    __size = 30
    __horizontal_space = 10
    __vertical_space = 25

    def __init__(self, deck):
        pygame.init()
        self.__screen = pygame.display.set_mode((self.__side_length, self.__side_length + self.__human_player_cards_menu_height))
        self.__init_images(deck)
        self.__num_players_to_positions = {i: [(self.__side_length // 2 + int(self.__radius * cos(radians(90 + j * (360 // i)))),
                                                self.__side_length // 2 - 35 + int(self.__radius * sin(radians(90 + j * (360 // i))))) for j in range(i)] for i in range(deck.total_num_cards)}
        self.__font = pygame.font.Font(pygame.font.get_default_font(), self.__size)

    def __init_images(self, deck):
        self.__card_images = dict()
        self.__card_size = None
        for card in deck.cards:
            if card[0] < deck.JACK:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", str(card[0]) + deck.STRING_RANKS[card[1]] + ".png"))
            elif card[0] == deck.JACK:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "J" + deck.STRING_RANKS[card[1]] + ".png"))
            elif card[0] == deck.QUEEN:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "Q" + deck.STRING_RANKS[card[1]] + ".png"))
            elif card[0] == deck.KING:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "K" + deck.STRING_RANKS[card[1]] + ".png"))
            else:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "A" + deck.STRING_RANKS[card[1]] + ".png"))
            if self.__card_size is None:
                self.__card_size = self.__card_images[card].get_size()
        self.__rank_images = dict()
        for rank in deck.RANKS:
            if rank == deck.HEARTS:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Hearts.png"))
            elif rank == deck.CLUBS:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Clubs.png"))
            elif rank == deck.DIAMONDS:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Diamonds.png"))
            elif rank == deck.SPADES:
                self.__rank_images[rank] = pygame.image.load(os.path.join("Images", "Spades.png"))
        self.__table_image = pygame.image.load(os.path.join("Images", "Table.png"))
        self.__back_of_card_image = pygame.image.load(os.path.join("Images", "ZCardBack.png"))
        self.__initial_card_position = (self.__side_length // 2 - 3 * (self.__card_size[0] + self.__horizontal_space), self.__side_length // 2 - (3 * self.__card_size[1]) // 2)
        self.__deck_initial_pos = (self.__side_length // 2 - self.__card_size[0] // 2, self.__side_length // 2 + self.__card_size[1] // 2)

    def show_screen(self, players, table, attacker, defender, deck, trump_rank):
        self.__screen.fill((0, 0, 0))
        self.__screen.blit(self.__table_image, (0, 0))
        self.__screen.blit(self.__rank_images[trump_rank], (0, 0))
        for i, position in enumerate(self.__num_players_to_positions[len(players)]):
            if players[i] == attacker:
                text = self.__font.render(players[i].name, True, (255, 0, 0))
            elif players[i] == defender:
                text = self.__font.render(players[i].name, True, (0, 0, 255))
            elif i == 0:
                text = self.__font.render(players[i].name, True, (255, 255, 255))
            else:
                text = self.__font.render(players[i].name, True, (0, 0, 0))
            rect = text.get_rect()
            rect.center = position
            self.__screen.blit(text, rect.topleft)
            if type(players[i]) == HumanPlayer:
                horizontal_space = 0
                if players[i].hand_size:
                    horizontal_space = (self.__side_length - 20) // players[i].hand_size
                for j, card in enumerate(players[i].hand):
                    self.__screen.blit(self.__card_images[card], (10 + j * horizontal_space, self.__side_length + 15))
            else:
                for j in range(players[i].hand_size):
                    self.__screen.blit(self.__back_of_card_image, (rect.left + j * 4, rect.bottom))
        for i in range(len(table)):
            for j in range(len(table[i])):
                self.__screen.blit(self.__card_images[table[i][j]], (self.__initial_card_position[0] + j * (self.__horizontal_space + self.__card_size[0]),
                                                                     self.__initial_card_position[1] + i * self.__vertical_space))
        for i in range(deck.current_num_cards):
            self.__screen.blit(self.__back_of_card_image, (self.__deck_initial_pos[0], self.__deck_initial_pos[1] - i))
        pygame.display.flip()
        pygame.time.wait(50)

    @staticmethod
    def end():
        pygame.quit()
