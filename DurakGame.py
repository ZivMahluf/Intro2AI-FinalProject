from DurakPlayer import DurakPlayer, BasePlayer, HumanPlayer
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
        self.__human_player_in_game = False
        self.__gui = GUI(Deck())

    def add_player(self, player):
        if len(self.__players) < self.MAX_PLAYERS:
            if type(player) == HumanPlayer and not self.__human_player_in_game:
                self.__human_player_in_game = True
                self.__constant_order_players.insert(0, player)
                self.__players.insert(0, player)
            elif type(player) != HumanPlayer:
                self.__constant_order_players.append(player)
                self.__players.append(player)

    def __reset_table(self):
        self.__attacking = list()
        self.__defending = list()
        self.__table = (self.__attacking, self.__defending)

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
            self.__gui.show_screen(self.__constant_order_players, self.__table, None, None, self.__deck, self.__trump_rank)
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
        self.__successfully_defended = False
        self.__attacking_limit = min(self.__players[self.__defender].hand_size, self.HAND_SIZE)
        self.__prev_table = tuple()
        self.__prev_action = Deck.NO_CARD
        self.__reward = 0

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
        for _ in range(attacker):
            self.__players.append(self.__players.pop(0))

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
        defending = True
        self.__successfully_defended = False
        self.__attacking_limit = min(self.__players[self.__defender].hand_size, self.HAND_SIZE)
        while defending and self.__playing:
            self.__check_events()
            if self.__playing:
                defending = self.__do_mini_round()
        if self.__playing:
            self.__end_round()

    def __check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__playing = False
                self.__quit_games = True

    def __do_mini_round(self):
        attacker_name, attacking_card = self.__attack()
        if attacking_card is None:
            self.__playing = False
            self.__quit_games = True
            defending = False
        elif attacking_card != Deck.NO_CARD:
            self.__attacking.append(attacking_card)
            self.__update_progress(attacker_name, attacking_card)
            defending = self.__defend()
        else:
            defending = False
            self.__successfully_defended = True
        if defending and len(self.__attacking) == self.__attacking_limit:
            defending = False
            self.__successfully_defended = True
        return defending

    def __attack(self):
        if self.__players[self.__attacker].hand_size:
            attacking_card = self.__players[self.__attacker].attack(self.__table, self.__get_legal_attack_cards(self.__players[self.__attacker]))
        else:
            attacking_card = Deck.NO_CARD
        attacker_name = self.__players[self.__attacker].name
        if attacking_card == Deck.NO_CARD:
            next_attacker = (self.__attacker + 1) % len(self.__players)
            while next_attacker != self.__defender:
                if self.__players[next_attacker].hand_size:
                    attacking_card = self.__players[next_attacker].attack(self.__table, self.__get_legal_attack_cards(self.__players[next_attacker]))
                    if attacking_card == Deck.NO_CARD:
                        next_attacker += 1
                    else:
                        attacker_name = self.__players[next_attacker].name
                        break
                else:
                    next_attacker += 1
        return attacker_name, attacking_card

    def __defend(self):
        defending_card = self.__players[self.__defender].defend(self.__table, self.__get_legal_defending_cards(self.__players[self.__defender]))
        if defending_card is None:
            self.__playing = False
            self.__quit_games = True
            defending = False
        elif defending_card != Deck.NO_CARD:
            self.__defending.append(defending_card)
            self.__update_progress(self.__players[self.__defender].name, defending_card)
            defending = True
        else:
            defending = False
        return defending

    def __throw_in(self):
        if len(self.__attacking) < self.__attacking_limit:
            adding_player = self.__attacker
            card_to_add = Deck.NO_CARD
            while adding_player != self.__defender:
                if self.__players[adding_player].hand_size:
                    card_to_add = self.__players[adding_player].attack(self.__table, self.__get_legal_attack_cards(self.__players[adding_player]))
                if card_to_add != Deck.NO_CARD:
                    self.__attacking.append(card_to_add)
                    self.__update_progress(self.__players[adding_player].name, card_to_add)
                else:
                    adding_player += 1
                    if adding_player == self.__defender:
                        break
                if len(self.__attacking) == self.__attacking_limit:
                    break

    def __update_progress(self, name, card):
        for player in self.__players:
            player.update_round_progress(name, card)
        self.__gui.show_screen(self.__constant_order_players, self.__table, self.__players[self.__attacker], self.__players[self.__defender], self.__deck, self.__trump_rank)

    def __end_round(self):
        if not self.__successfully_defended:
            self.__throw_in()
            self.__players[self.__defender].take_cards(self.__defending + self.__attacking)
        self.__reset_table()
        self.__update_and_draw_cards()
        self.__remove_winners()
        self.__defender = len(self.__players) - 1
        if self.MIN_PLAYERS <= len(self.__players):
            self.__update_attacker_defender()
            self.__gui.show_screen(self.__constant_order_players, self.__table, self.__players[self.__attacker], self.__players[self.__defender], self.__deck, self.__trump_rank)
        else:
            self.__end_current_game()

    def __update_and_draw_cards(self):
        for player in self.__players:
            player.update_end_round(self.__players[self.__defender].name, self.__table, self.__successfully_defended)
            player.take_cards(self.__deck.draw(max(self.HAND_SIZE - player.hand_size, 0)))
            if player.hand_size == 0:
                self.__out_players.append(player)

    def __remove_winners(self):
        for out_player in self.__out_players:
            if out_player in self.__players:
                self.__players.remove(out_player)

    def __update_attacker_defender(self):
        self.__players.insert(self.__attacker, self.__players.pop(self.__defender))
        if not self.__successfully_defended:
            # If the defender failed to defend, the next attacker is the player that would be the next defender.
            self.__players.insert(self.__attacker, self.__players.pop(self.__defender))

    def __end_current_game(self):
        if len(self.__players):
            self.__losers.append(self.__players[0])
            print(self.__players[0].name, "lost!")
        else:
            self.__losers.append(None)
            print("Game ended in a " + str(len(self.__constant_order_players)) + "-way draw!")
        self.__playing = False

    def __get_legal_attack_cards(self, attacker: DurakPlayer):
        if len(self.__table[0]) == 0:
            return attacker.hand
        legal_attacking_cards = [Deck.NO_CARD]
        for card in attacker.hand:
            for i in range(len(self.__table)):
                for card_on_table in self.__table[i]:
                    if card[0] == card_on_table[0]:
                        legal_attacking_cards.append(card)
                        break
        return legal_attacking_cards

    def __get_legal_defending_cards(self, defender: DurakPlayer):
        legal_defending_cards = [Deck.NO_CARD]
        attacking_card = self.__table[0][-1]
        for card in defender.hand:
            if attacking_card[1] == card[1]:
                if attacking_card[0] < card[0]:
                    legal_defending_cards.append(card)
            elif (attacking_card[1] != self.__trump_rank) and (card[1] == self.__trump_rank):
                legal_defending_cards.append(card)
        return legal_defending_cards

    @property
    def trump_rank(self):
        return self.__trump_rank

    @property
    def losers(self):
        return [loser.name for loser in self.__losers if loser is not None]

    @property
    def gui(self):
        return self.__gui


games = 1
game = DurakGame()
game.add_player(BasePlayer(game.HAND_SIZE, "Ziv"))
game.add_player(BasePlayer(game.HAND_SIZE, "Idan"))
game.add_player(HumanPlayer(game.HAND_SIZE, "Vitaly", game.gui))
game.add_player(BasePlayer(game.HAND_SIZE, "Eyal"))
game.add_player(BasePlayer(game.HAND_SIZE, "Yoni"))
game.add_player(BasePlayer(game.HAND_SIZE, "Jeff"))
game.play_games(games)
