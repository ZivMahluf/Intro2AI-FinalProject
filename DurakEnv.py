from Types import List, Tuple, NumberType, StateType
from DurakPlayer import DurakPlayer
from HumanPlayer import HumanPlayer
from Deck import Deck
from GUI import GUI

import random


class DurakEnv:

    """
    Game Parameters.
    """
    MIN_PLAYERS = 2
    MAX_PLAYERS = 6
    HAND_SIZE = 6

    def __init__(self, players: List[DurakPlayer], render=False):
        """
        Constructor.
        :param players: List of players which will be playing the game. Between 2 and 6 players, and at most one human player.
        :param render: Weather to render the game of not.
        """
        # non-changing parameters:
        self.players = players
        self.to_render = render or (True in [isinstance(player, HumanPlayer) for player in players])  # if a human player is given, always render
        self.attacker = 0
        self.defender = 1
        self.trump_suit = self.deck.HEARTS
        # changing parameters (will be re-initialized upon calling the reset method):
        self.active_players = players[:]
        self.attacking_player = None
        self.turn_player = None
        self.loser = None
        self.attacking_cards = list()
        self.defending_cards = list()
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        self.gui = GUI() if self.to_render else None
        self.deck = Deck()
        self.defending = True
        self.successful = False
        self.limit = 0
        self.last_action = Deck.NO_CARD
        self.reset_round = False
        self.attack_phase = True
        self.reset_attacker = True
        self.reward = 0
        self.first_initialize_players()

    def first_initialize_players(self):
        """
        Initialization of
        :return:
        """
        for player in self.players:
            player.set_trump_suit(self.trump_suit)

    def reset(self) -> StateType:
        """
        Resets the environment for the game.
        :return: The initial state of the game.
        """
        random.shuffle(self.players)
        self.active_players = self.players[:]
        self.attacking_player = None
        self.turn_player = None
        self.loser = None
        self.attacking_cards = list()
        self.defending_cards = list()
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        self.gui = GUI() if self.to_render else None
        self.deck = Deck()
        self.defending = True
        self.successful = False
        self.limit = 0
        self.last_action = Deck.NO_CARD
        self.reset_round = False
        self.attack_phase = True
        self.reset_attacker = True
        self.reward = 0
        self.initialize_players()
        self.set_first_attacker()
        self.do_reset_round()
        self.update_legal_cards()
        players_hands_sizes = [player.hand_size for player in self.players]
        return self.attacking_cards[:], self.defending_cards[:], self.legal_attacking_cards[:], self.legal_defending_cards[:], self.deck.current_num_cards, players_hands_sizes

    def initialize_players(self) -> None:
        """
        Initializes the player for the game.
        """
        # Set GUI and initialize players for the game.
        for player in self.active_players:
            player.set_gui(self.gui)
            player.initialize_for_game()
        # Dealing starting hands
        deal = True
        while deal:
            self.initialize_deck()
            self.reset_hands()
            deal = self.deal_cards()

    def initialize_deck(self) -> None:
        """
        Resets and Shuffles the deck.
        """
        self.deck = Deck()
        self.deck.shuffle()

    def reset_hands(self) -> None:
        """
        Empties the players' hands.
        """
        for player in self.active_players:
            player.empty_hand()

    def deal_cards(self) -> bool:
        """
        Deals cards to the players, and checks weather their opening hands are legal.
        :return: True if a player has an illegal opening hand, False otherwise.
        """
        for player in self.active_players:
            drawn_cards = self.deck.draw(self.HAND_SIZE)
            player.take_cards(drawn_cards)
            if not player.is_starting_hand_legal():
                return True
        return False

    def set_first_attacker(self) -> None:
        """
        Determines the first attacker for the game.
        """
        lowest_trump_value = self.deck.ACE + 1
        starting_player_name = None
        for player in self.active_players:
            value = player.get_lowest_trump()
            if value < lowest_trump_value:
                lowest_trump_value = value
                starting_player_name = player.name
        if starting_player_name is not None:
            while self.active_players[self.attacker].name != starting_player_name:
                temp = self.active_players.pop(self.attacker)
                self.active_players.append(temp)

    def step(self, action) -> Tuple[StateType, NumberType, bool]:
        """
        Preforms a step from the current state to the next state of the game determined by the given action.
        :param action: The action to do.
        :return: The next state, a reward for the action, and weather the game is over.
        """
        self.last_action = action
        self.reward = 0
        if self.attack_phase:
            self.do_attack_phase()
        else:
            self.do_defence_phase()
        self.calculate_reward()
        self.check_end_round()
        if self.reset_round:
            self.update_end_round_players()
            self.update_players_hands()
            self.remove_winners()
            self.update_active_players_order()
            if not self.game_over():
                self.do_reset_round()
        self.update_legal_cards()
        self.dispose_events()
        players_hands_sizes = [player.hand_size for player in self.players]
        return (self.attacking_cards[:], self.defending_cards[:], self.legal_attacking_cards[:], self.legal_defending_cards[:], self.deck.current_num_cards, players_hands_sizes), \
            self.reward, self.game_over()

    def do_attack_phase(self) -> None:
        """
        Preforms an action in the attack phase, and updates the state and players.
        """
        if self.last_action != self.deck.NO_CARD:
            self.attacking_cards.append(self.last_action)
            self.attack_phase = False
        else:
            if self.turn_player == self.active_players[-1]:
                # last player with a chance to attack chose not to, so the round ends with a successful defence.
                self.defending = False
                self.successful = True
                self.reset_attacker = True
            else:
                # another player might have a chance to attack next
                self.reset_attacker = False
                self.turn_player = self.active_players[self.active_players.index(self.turn_player) + 1]  # next player to the left
                if self.turn_player == self.active_players[self.defender]:
                    # the next player was the defender (meaning, the original attacker chose to pass)
                    if self.turn_player != self.active_players[-1]:
                        # there is another player after the defender who might be able to attack and is not the original attacker.
                        self.turn_player = self.active_players[self.active_players.index(self.turn_player) + 1]
                    else:
                        # there are no more players who might choose to attack, so the round ends with a successful defence
                        self.turn_player = self.active_players[self.active_players.index(self.turn_player) - 1]
                        self.defending = False
                        self.successful = True
                        self.reset_attacker = True

    def do_defence_phase(self) -> None:
        """
        Preforms an action in the defence phase, and updates the state and players.
        """
        if self.last_action != self.deck.NO_CARD:
            self.defending_cards.append(self.last_action)
            self.attack_phase = True
            if len(self.defending_cards) == self.limit:
                # successful defence
                self.defending = False
                self.successful = True
            self.reset_attacker = True
        else:
            # defence failed
            self.defending = False
            self.successful = False
            self.turn_player.take_cards(self.attacking_cards)
            self.turn_player.take_cards(self.defending_cards)

    def update_players_hands(self) -> None:
        """
        Each player draws until holding 6 cards or til the deck is empty.
        """
        for player in self.active_players[:self.defender] + self.active_players[self.defender + 1:]:
            drawn_cards = self.deck.draw(max(0, self.HAND_SIZE - player.hand_size))
            player.take_cards(drawn_cards)
        drawn_cards = self.deck.draw(max(0, self.HAND_SIZE - self.active_players[self.defender].hand_size))
        self.active_players[self.defender].take_cards(drawn_cards)

    def remove_winners(self) -> None:
        """
        Removes winners (players who have no cards in hand when the deck is empty) from the game.
        """
        to_remove = list()
        for player in self.active_players:
            if player.hand_size == 0:
                to_remove.append(player)
        for player in to_remove:
            self.active_players.remove(player)

    def update_active_players_order(self) -> None:
        """
        Updates the order of the players still in the game.
        """
        if len(self.active_players) == 1:
            self.loser = self.active_players[0]
        elif len(self.active_players) >= self.MIN_PLAYERS:
            if self.active_players[self.attacker] == self.attacking_player:
                # This condition checks if the attacking player is still in the game
                self.move_attacker_back()
            if not self.successful:
                # here there is no need to check if the defender is in the game since in this case
                # the defender took all cards, and is necessarily in the game.
                self.move_attacker_back()

    def move_attacker_back(self) -> None:
        """
        Moves the current attacker to the end of the active players list.
        """
        temp = self.active_players.pop(self.attacker)
        self.active_players.append(temp)

    def do_reset_round(self) -> None:
        """
        Resets the parameters of the round.
        """
        self.attacking_cards = list()
        self.defending_cards = list()
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        self.turn_player = self.active_players[self.attacker]
        self.attacking_player = self.active_players[self.attacker]
        self.defending = True
        self.successful = False
        self.limit = min(self.HAND_SIZE, self.active_players[self.defender].hand_size)
        self.attack_phase = True
        self.reset_attacker = True
        self.reset_round = False

    def update_legal_cards(self) -> None:
        """
        Updates the lists of legal attack and defence cards.
        """
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        if len(self.attacking_cards) == 0:
            self.legal_attacking_cards = self.deck.get_full_list_of_cards()
            self.legal_defending_cards = [Deck.NO_CARD]
        else:
            self.legal_attacking_cards = [self.deck.NO_CARD]
            self.legal_defending_cards = [self.deck.NO_CARD]
            for card in self.deck.get_full_list_of_cards():
                if card not in self.attacking_cards and card not in self.defending_cards:
                    for a_card in self.attacking_cards:
                        if card[0] == a_card[0] and card[1] != a_card[1]:
                            self.legal_attacking_cards.append(card)
                            break
                    if card not in self.legal_attacking_cards:
                        for d_card in self.defending_cards:
                            if card[0] == d_card[0] and card[1] != d_card[1]:
                                self.legal_attacking_cards.append(card)
                                break
                if len(self.attacking_cards) > len(self.defending_cards):
                    if card not in self.defending_cards:
                        if (card[1] == self.attacking_cards[-1][1] and card[0] > self.attacking_cards[-1][0]) or \
                                ((card[1] == self.trump_suit) and (self.attacking_cards[-1][1] != self.trump_suit)):
                            self.legal_defending_cards.append(card)

    def dispose_events(self) -> None:
        """
        Disposes accumulated events from the GUI.
        """
        self.gui.dispose_events() if self.gui is not None else None

    def calculate_reward(self) -> None:
        """
        Calculates the reward for the action.
        """
        if not self.turn_player.hand_size and not self.deck.current_num_cards:
            self.reward = 30
        elif not self.defending:
            if self.turn_player == self.active_players[self.defender]:
                if self.successful:
                    self.reward = 1
                else:
                    self.reward = -1

    def check_end_round(self) -> None:
        """
        Checks weather the round is over.
        If not, updates the turn player.
        """
        if not self.defending:
            self.reset_round = True
        else:
            if self.attack_phase and self.reset_attacker:
                self.turn_player = self.active_players[self.attacker]
            elif not self.attack_phase:
                self.turn_player = self.active_players[self.defender]

    def update_end_round_players(self) -> None:
        """
        Updates the players regarding the result of the round.
        """
        for player in self.active_players:
            player.update_end_round(self.active_players[self.defender].name, (self.attacking_cards, self.defending_cards), self.successful)

    def render(self) -> None:
        """
        Renders the current state of the game.
        """
        if self.to_render and self.gui is not None:
            attacker = self.active_players[self.attacker] if len(self.active_players) else None
            defender = self.active_players[self.defender] if len(self.active_players) >= self.MIN_PLAYERS else None
            self.gui.show_screen(self.players, (self.attacking_cards, self.defending_cards),
                                 attacker, defender,
                                 self.deck, self.trump_suit)

    def game_over(self) -> bool:
        """
        :return: True if the game is over.
        """
        return len(self.active_players) < self.MIN_PLAYERS

    def get_turn_player(self) -> DurakPlayer:
        """
        :return: The player whose turn it is.
        """
        return self.turn_player

    def to_attack(self) -> bool:
        """
        :return: Weather the turn player should attack.
        """
        return self.attack_phase

    def end_gui(self) -> None:
        """
        Ends the run of the GUI.
        """
        self.gui.end() if self.gui is not None else None

    def get_loser(self):
        """
        :return: The loser of the game (None if the game is not over yet, or there was no loser).
        """
        return self.loser

    def get_available_actions(self):
        """
        The available actions are chosen by checking whether or not the player is attacking, and then taking the legal attacking/defending cards
        of the player
        :return: The available actions (cards) of the player. empty set if no available actions found
        """
        if self.turn_player == self.active_players[self.defender]:
            return set(self.turn_player.hand).intersection(set(self.legal_defending_cards))
        return set(self.turn_player.hand).intersection(set(self.legal_attacking_cards))
