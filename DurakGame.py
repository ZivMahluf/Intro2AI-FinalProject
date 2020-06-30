from DurakPlayer import BasePlayer
from Deck import Deck
from GUI import GUI
import numpy as np
import pygame


class DurakGame:

    MIN_PLAYERS = 2
    HAND_SIZE = 6
    MAX_PLAYERS = 6

    def __init__(self):
        self.__constant_order_players = list()
        self.__players = list()
        self.__losers = list()
        self.__quit_games = False
        self.__gui = GUI(Deck())

    def add_player(self, player):
        if len(self.__players) < self.MAX_PLAYERS:
            self.__players.append(player)
            self.__constant_order_players.append(player)

    def __reset_table(self):
        self.__attacking = list()
        self.__defending = list()
        self.__table = [self.__attacking, self.__defending]

    def play_games(self, num_games=1):
        for g in range(num_games):
            if self.__quit_games:
                break
            print("Game", g + 1, "of", num_games)
            self.__initialize_deck()
            self.__initialize_game_parameters()
            self.__deal_hands()
            self.__check_hands()
            self.__set_first()
            while self.__playing:
                self.__do_round()
            self.__players = self.__players + self.__out_players
        self.__gui.end()

    def __initialize_deck(self):
        self.__deck = Deck()
        self.__trump_rank = np.random.choice([self.__deck.HEARTS, self.__deck.CLUBS, self.__deck.DIAMONDS, self.__deck.SPADES])
        for player in self.__players:
            player.empty_hand()
            player.set_trump_rank(self.__trump_rank)

    def __initialize_game_parameters(self):
        self.__attacker = 0
        self.__defender = len(self.__players) - 1
        self.__playing = True
        self.__reset_table()
        self.__out_players = list()
        # self.__prev_table = []
        # self.__prev_action = Deck.NO_CARD
        # self.__reward = 0

    def __deal_hands(self):
        for player in self.__players:
            player.take_cards(self.__deck.draw(self.HAND_SIZE))

    def __check_hands(self):
        all_legal = False
        while not all_legal:
            all_legal = True
            for player in self.__players:
                if not player.is_starting_hand_legal():
                    all_legal = False
            if not all_legal:
                self.__initialize_deck()
                self.__deal_hands()

    def __set_first(self):
        min_value = Deck.ACE
        attacker = 0
        for i, player in enumerate(self.__players):
            value = player.get_lowest_trump()
            if Deck.NO_CARD < value < min_value:
                attacker = i
                min_value = value
        self.__players.insert(self.__attacker, self.__players.pop(attacker))

    def __do_round(self):
        """
        General principles:
        1. The order of attacking priority is:
            1. attacker
            2. the players after the defender in increasing cyclical order
            * example (4 players - 0, 1, 2, 3):
                attacker = 1,
                defender = 2,
                if the attacker does not attack, 3 is asked,
                if 3 doesn't attack, 0 is asked.
        2. If the defender takes all cards on the table and it would be possible to attack more had the player not taken the cards:
            - every other player gets to add cards for the defending player to take up to the limit of attacking cards, in order of priority
        Round description:
        get first attack from attacker (non optional)
        While
        1. The total number of attacking cards is at most min(defenders hand size at the start of the round, initial hand size)
        AND
        2. The number of unanswered attacking cards is 1
        :
            - get response card from the defender
                 - no response card means taking all cards on the table
                 - in case of taking:
                    - if the total number of attacking cards is strictly less than min(defenders hand size at the start of the round, initial hand size):
                        - while the above condition holds:
                            - get another attacking card (if possible)
            - if not taken:
                - get next attacking card
        """
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         self.__playing = False
        #         self.__quit_games = True
        #         return
        defending = True
        successfully_defended = False
        attacking_limit = min(self.__players[self.__defender].hand_size, self.HAND_SIZE)
        while defending:
            # self.__gui.show_screen(self.__constant_order_players, self.__table)
            # attacking
            if len(self.__attacking) == 0:
                # it is forbidden that the attacker doesn't put a card if the table is empty.
                attacking_card = self.__players[self.__attacker].attack(self.__table)
            else:
                attacking_card = self.__players[self.__attacker].attack(self.__table)
                if attacking_card == Deck.NO_CARD:
                    next_attacker = (self.__attacker + 1) % len(self.__players)
                    while next_attacker != self.__defender:
                        if self.__players[next_attacker].hand_size:
                            attacking_card = self.__players[next_attacker].attack(self.__table)
                            if attacking_card == Deck.NO_CARD:
                                next_attacker += 1
                            else:
                                break
                        else:
                            next_attacker += 1
            if attacking_card != Deck.NO_CARD:
                self.__attacking.append(attacking_card)
                # defending
                defending_card = self.__players[self.__defender].defend(self.__table)
                if defending_card == Deck.NO_CARD:
                    # If the defender took the cards, then the attackers have the option to add more cards, up to the attacking limit
                    if len(self.__attacking) < attacking_limit:
                        adding_player = self.__attacker
                        card_to_add = Deck.NO_CARD
                        while adding_player != self.__defender:
                            if self.__players[adding_player].hand_size:
                                card_to_add = self.__players[adding_player].attack(self.__table)
                            if card_to_add != Deck.NO_CARD:
                                self.__attacking.append(card_to_add)
                            else:
                                adding_player += 1
                                if adding_player == self.__defender:
                                    break
                            if len(self.__attacking) == attacking_limit:
                                break
                    self.__players[self.__defender].take_cards(self.__defending)
                    self.__players[self.__defender].take_cards(self.__attacking)
                    defending = False
                else:
                    self.__defending.append(defending_card)
            else:
                defending = False
                successfully_defended = True
            if defending and len(self.__attacking) == attacking_limit:
                defending = False
                successfully_defended = True
        # clearing the table
        self.__reset_table()
        # dealing cards as needed and eliminating winning players
        for player in self.__players:
            player.take_cards(self.__deck.draw(max(self.HAND_SIZE - player.hand_size, 0)))
            if player.hand_size == 0:
                self.__out_players.append(player)
        for out_player in self.__out_players:
            if out_player in self.__players:
                self.__players.remove(out_player)
        self.__defender = len(self.__players) - 1
        if self.MIN_PLAYERS <= len(self.__players):
            self.__players.insert(self.__attacker, self.__players.pop(self.__defender))
            if not successfully_defended:
                # If the defender failed to defend, the next attacker is the player that would be the next defender.
                self.__players.insert(self.__attacker, self.__players.pop(self.__defender))
        else:
            if len(self.__players):
                self.__losers.append(self.__players[0])
            else:
                self.__losers.append(None)
            self.__playing = False

    @property
    def trump_rank(self):
        return self.__trump_rank

    @property
    def losers(self):
        return [loser.name for loser in self.__losers if loser is not None]


games = 5000
game = DurakGame()
game.add_player(BasePlayer(game.HAND_SIZE, "David"))
game.add_player(BasePlayer(game.HAND_SIZE, "Arkady"))
game.add_player(BasePlayer(game.HAND_SIZE, "Vitaly"))
game.add_player(BasePlayer(game.HAND_SIZE, "Sveta"))
game.add_player(BasePlayer(game.HAND_SIZE, "Alex"))
game.add_player(BasePlayer(game.HAND_SIZE, "Eli"))
game.play_games(games)
