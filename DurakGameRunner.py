from DurakPlayer import DurakPlayer
from HumanPlayer import HumanPlayer
from GUI import GUI
from DurakGameLogic import DurakLogic
from Deck import Deck


class DurakRunner:

    MIN_PLAYERS = 2
    MAX_PLAYERS = 6
    HAND_SIZE = 6

    def __init__(self):
        self.players = list()
        self.human_player_exists = False
        self.render = self.human_player_exists  # if a human player is playing, always render
        self.verbose = False
        self.gui = GUI() if self.render else None
        self.attacking_cards = list()
        self.defending_cards = list()
        self.table = (self.attacking_cards, self.defending_cards)
        self.attacker = 0
        self.defender = 1
        self.active_players = list()
        self.losers = list()
        self.first_game = True
        self.deck = Deck()
        self.trump_rank = Deck.HEARTS
        self.game_logic = DurakLogic()
        self.game = 0
        self.round = 0
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        self.attacking_card = Deck.NO_CARD
        self.defending_card = Deck.NO_CARD
        self.last_attacker_name = ""
        self.attacking_player = None
        self.successful = False
        self.defending = False
        self.limit = 0

    def add_player(self, player: DurakPlayer) -> None:
        """
        Adds a new player.
        Only one HumanPlayer object is allowed. Any following HumanPlayer will not be added.
        :param player: Player to add.
        """
        if (type(player) == HumanPlayer and not self.human_player_exists) or (type(player) != HumanPlayer):
            if len(self.players) < self.MAX_PLAYERS:
                self.first_game = True
                self.players.append(player)
                if type(player) == HumanPlayer:
                    self.human_player_exists = True
                    self.render = True

    def play_games(self, games: int = 1, render: bool = True, verbose: bool = False) -> None:
        """
        Plays multiple consecutive games of durak with the players added.
        :param games: Number of games to play.
        :param render: Weather of not to render the game (if a HumanPlayer participates, the games will be rendered regardless).
        :param verbose: Weather to print a progress of the game.
        """
        self.render = render or self.human_player_exists
        self.verbose = verbose
        if len(self.players) >= self.MIN_PLAYERS:
            for self.game in range(1, games + 1):
                self.play_game()

    def play_game(self) -> None:
        """
        Plays a single full game.
        """
        if self.verbose:
            print('---------------------', 'game', self.game, '---------------------')
        self.initialize_game()
        self.round = 1
        if self.render:
            self.gui.show_screen(self.players, (list(), list()), None, None, self.deck, self.trump_rank)
        while not self.game_over():
            self.play_round()
        self.first_game = False
        if self.render:
            self.gui.show_screen(self.players, (list(), list()), None, None, self.deck, self.trump_rank)

    def initialize_game(self) -> None:
        """
        Initializes a new game.
        """
        self.gui = GUI() if self.render else None
        self.initialize_table()
        self.initialize_players()
        self.choose_starting_player()

    def initialize_table(self) -> None:
        """
        Initializes an empty table.
        """
        self.attacking_cards = list()
        self.defending_cards = list()
        self.table = (self.attacking_cards, self.defending_cards)

    def initialize_players(self) -> None:
        """
        Initializes the player objects before the game.
        """
        self.active_players = self.players[:]
        for player in self.active_players:
            player.set_gui(self.gui)
        deal = True
        while deal:
            self.initialize_deck()
            self.reset_hands()
            deal = self.deal_cards()

    def initialize_deck(self) -> None:
        """
        Initializes a new deck for the game.
        """
        self.deck = Deck()
        self.deck.shuffle()
        trump_card = self.deck.draw()[0]
        self.trump_rank = trump_card[1]
        self.deck.to_bottom(trump_card)

    def reset_hands(self) -> None:
        """
        Empties the hands of all players, and sets the current trump rank for them.
        """
        for player in self.active_players:
            player.empty_hand()
            player.set_trump_rank(self.trump_rank)

    def deal_cards(self) -> bool:
        """
        Deals opening hands to the players.
        :return: Weather any hand dealt if illegal.
        """
        for player in self.active_players:
            drawn_cards = self.deck.draw(self.HAND_SIZE)
            player.take_cards(drawn_cards)
            if not player.is_starting_hand_legal():
                return True
        return False

    def choose_starting_player(self) -> None:
        """
        Chooses a player to start the next game.
        """
        if self.first_game or self.losers[-1] is None:
            self.choose_starting_player_by_card()
        else:
            self.choose_starting_player_by_last_loser()

    def choose_starting_player_by_card(self) -> None:
        """
        Chooses the player with the lowest valued trump card to attack first. If not player has a trump card, then the first player in the active players list starts.
        """
        lowest_trump_value = Deck.ACE + 1
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

    def choose_starting_player_by_last_loser(self) -> None:
        """
        Chooses a player which will attack first such that the loser of the last game is the first to be attacked.
        """
        while self.active_players[self.defender].name != self.losers[-1].name:
            temp = self.active_players.pop(self.attacker)
            self.active_players.append(temp)

    def game_over(self) -> bool:
        """
        :return: Weather the game is over.
        """
        return len(self.active_players) < self.MIN_PLAYERS

    def play_round(self) -> None:
        """
        Runs a single full round of the game.
        """
        if self.verbose:
            print('round', self.round, '(game ' + str(self.game) + ')')
            self.round += 1
        self.reset_round_parameters()
        if self.render:
            self.gui.show_screen(self.players, self.table, self.active_players[self.attacker], self.active_players[self.defender], self.deck, self.trump_rank)
        while not self.check_quit_from_gui() and self.defending:
            self.do_attack_phase()
            self.do_defence_phase()
        if not self.successful:
            self.do_scoop_phase()
        self.refill_hands()
        self.remove_winners()
        self.check_end_game()

    def reset_round_parameters(self) -> None:
        """
        Resets the required parameters for each round.
        """
        self.initialize_table()
        self.attacking_player = self.active_players[self.attacker]
        self.successful = False
        self.defending = True
        self.limit = min(self.HAND_SIZE, self.active_players[self.defender].hand_size)

    def check_quit_from_gui(self) -> bool:
        """
        :return: Weather the user pressed the 'x' button on the GUI window (this will end the entire sequence of games).
        """
        if self.render:
            if self.gui.pressed_quit():
                self.active_players = list()
                return True
        return False

    def do_attack_phase(self) -> None:
        """
        One of the attacking players attacks with a card (in the order in the active players list).
        """
        self.legal_attacking_cards = self.game_logic.get_legal_attacking_cards(self.active_players[self.attacker].hand, self.table)
        self.attacking_card = self.active_players[self.attacker].attack(self.table, self.legal_attacking_cards)
        self.last_attacker_name = self.active_players[self.attacker].name
        if self.attacking_card == Deck.NO_CARD:
            for player in self.active_players[self.defender + 1:]:
                self.legal_attacking_cards = self.game_logic.get_legal_attacking_cards(player.hand, self.table)
                self.attacking_card = player.attack(self.table, self.legal_attacking_cards)
                self.last_attacker_name = player.name
                if self.attacking_card != Deck.NO_CARD:
                    return

    def do_defence_phase(self) -> None:
        """
        The defending player defends as needed from the last attacking card.
        """
        if self.attacking_card == Deck.NO_CARD:
            self.successful = True
            self.defending = False
            return
        else:
            self.attacking_cards.append(self.attacking_card)
            if self.render:
                self.gui.show_screen(self.players, self.table, self.active_players[self.attacker], self.active_players[self.defender], self.deck, self.trump_rank)
            self.legal_defending_cards = self.game_logic.get_legal_defending_cards(self.active_players[self.defender].hand, self.attacking_card, self.trump_rank)
            self.defending_card = self.active_players[self.defender].defend(self.table, self.legal_defending_cards)
            if self.defending_card == Deck.NO_CARD:
                self.defending = False
                return
            else:
                self.defending_cards.append(self.defending_card)
                if self.render:
                    self.gui.show_screen(self.players, self.table, self.active_players[self.attacker], self.active_players[self.defender], self.deck, self.trump_rank)
        if len(self.attacking_cards) == len(self.defending_cards) == self.limit:
            self.successful = True
            self.defending = False

    def do_scoop_phase(self) -> None:
        """
        The defending player takes all cards on the table (possibly with the addition of more from other players).
        """
        if len(self.attacking_cards) < self.limit:
            self.do_throw_in_phase()
        self.active_players[self.defender].take_cards(self.attacking_cards)
        self.active_players[self.defender].take_cards(self.defending_cards)

    def do_throw_in_phase(self) -> None:
        """
        Each player that's not defending has a chance to throw as many cards as allowed for the defender to take.
        """
        for player in self.active_players[:self.defender] + self.active_players[self.defender + 1:]:
            self.legal_attacking_cards = self.game_logic.get_legal_attacking_cards(player.hand, self.table)
            self.attacking_card = player.attack(self.table, self.legal_attacking_cards)
            self.last_attacker_name = player.name
            while self.attacking_card != Deck.NO_CARD and len(self.attacking_cards) < self.limit:
                self.attacking_cards.append(self.attacking_card)
                self.legal_attacking_cards.remove(self.attacking_card)
                self.attacking_card = player.attack(self.table, self.legal_attacking_cards)
                if self.render:
                    self.gui.show_screen(self.players, self.table, self.active_players[self.attacker], self.active_players[self.defender], self.deck, self.trump_rank)

    def refill_hands(self) -> None:
        """
        Deals each player cards from the deck until the player has HAND_SIZE cards in the hand or the deck is empty.
        """
        for player in self.active_players[:self.defender] + self.active_players[self.defender + 1:]:
            drawn_cards = self.deck.draw(max(0, self.HAND_SIZE - player.hand_size))
            player.take_cards(drawn_cards)
        drawn_cards = self.deck.draw(max(0, self.HAND_SIZE - self.active_players[self.defender].hand_size))
        self.active_players[self.defender].take_cards(drawn_cards)

    def remove_winners(self) -> None:
        """
        Removes the players with no cards in hand from the list of active players.
        """
        to_remove = list()
        for player in self.active_players:
            if player.hand_size == 0:
                to_remove.append(player)
        for player in to_remove:
            self.active_players.remove(player)

    def check_end_game(self) -> None:
        """
        Checks weather the game is over. If yes - updates the loser (if any), otherwise - updates the attacker for the next round.
        """
        if len(self.active_players) == 0:
            self.losers.append(None)  # NO LOSER
        elif len(self.active_players) == 1:
            if self.verbose:
                print(self.active_players[0].name, "lost after", self.round, 'rounds')
            self.losers.append(self.active_players[0])  # the remaining player is the loser
        elif len(self.active_players) >= self.MIN_PLAYERS:
            self.update_attacker()

    def update_attacker(self) -> None:
        """
        Updates the attacker for the next round.
        """
        if self.active_players[self.attacker] == self.attacking_player:
            self.move_attacker_back()
        if not self.successful:
            self.move_attacker_back()

    def move_attacker_back(self) -> None:
        """
        Moves the current attacker to the end of the active players list.
        """
        temp = self.active_players.pop(self.attacker)
        self.active_players.append(temp)

    def end(self) -> None:
        """
        Quits the GUI (if exists).
        """
        self.gui.end() if self.gui is not None else None
