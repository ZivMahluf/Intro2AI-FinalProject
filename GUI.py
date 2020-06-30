import pygame
import os
from math import sin, cos, radians


class GUI:

    __side_length = 750
    __radius = __side_length // 2.3
    __size = 30

    def __init__(self, deck):
        pygame.init()
        # self.__screen = pygame.display.set_mode((self.__side_length, self.__side_length))
        self.__deck = deck
        self.__init_images()
        self.__num_players_to_positions = {i: [(self.__side_length // 2 + int(self.__radius * cos(radians(j * (360 // i)))),
                                                self.__side_length // 2 + int(self.__radius * sin(radians(j * (360 // i))))) for j in range(i)] for i in range(deck.total_num_cards)}
        # self.__font = pygame.font.Font(pygame.font.get_default_font(), self.__size)

    def __init_images(self):
        self.__card_images = dict()
        for card in self.__deck.cards:
            if card[0] < self.__deck.JACK:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", str(card[0]) + card[1] + ".png"))
            elif card[0] == self.__deck.JACK:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "J" + card[1] + ".png"))
            elif card[0] == self.__deck.QUEEN:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "Q" + card[1] + ".png"))
            elif card[0] == self.__deck.KING:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "K" + card[1] + ".png"))
            else:
                self.__card_images[card] = pygame.image.load(os.path.join("Images", "A" + card[1] + ".png"))
        self.__table_image = pygame.image.load(os.path.join("Images", "Table.png"))
        self.__back_of_card_image = pygame.image.load(os.path.join("Images", "ZCardBack.png"))

    def show_screen(self, players, table):
        self.__screen.blit(self.__table_image, (0, 0))
        for i, position in enumerate(self.__num_players_to_positions[len(players)]):
            text = self.__font.render(players[i].name, True, (0, 0, 0))
            rect = text.get_rect()
            rect.center = position
            self.__screen.blit(text, rect.topleft)
            for j in range(players[i].hand_size):
                self.__screen.blit(self.__back_of_card_image, (rect.left + j * 4, rect.bottom))
        pygame.display.flip()
        pygame.time.wait(20)

    @staticmethod
    def end():
        pygame.quit()
