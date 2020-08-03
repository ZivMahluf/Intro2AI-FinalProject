from DurakPlayer import DurakPlayer
from HumanPlayer import HumanPlayer
from RandomPlayer import RandomPlayer
from BasicPlayer import BasicPlayer
from LearningPlayer import LearningPlayer
from NFSPPlayer import NFSPPlayer
from typing import List, Tuple, Union, Optional
from Deck import Deck
from GUI import GUI


class DurakEnv:

    MIN_PLAYERS = 2
    MAX_PLAYERS = 6
    HAND_SIZE = 6

    CardListType = List[Deck.CardType]
    StateType = Tuple[CardListType, CardListType, CardListType, CardListType]
    NumberType = Union[int, float]
    InfoType = List

    def __init__(self, players: List[DurakPlayer], render=False):
        # non-changing parameters:
        self.players = players
        self.to_render = render or (
            True in [isinstance(player, HumanPlayer) for player in players])
        self.attacker = 0
        self.defender = 1
        # changing parameters (will be re-initialized upon calling the reset method):
        self.active_players = players
        self.attacking_player = None
        self.turn_player = None
        self.loser = None
        self.attacking_cards = list()
        self.defending_cards = list()
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        self.state = (self.attacking_cards, self.defending_cards,
                      self.legal_attacking_cards, self.legal_defending_cards)
        self.gui = GUI() if self.to_render else None
        self.deck = Deck()
        self.trump_rank = self.deck.HEARTS
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
        players_names = [player.name for player in self.players]
        for player in self.players:
            player.first_initialize(players_names, self.deck.total_num_cards)

    def reset(self) -> StateType:
        self.active_players = self.players[:]
        self.attacking_player = None
        self.turn_player = None
        self.loser = None
        self.attacking_cards = list()
        self.defending_cards = list()
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        self.state = (self.attacking_cards, self.defending_cards,
                      self.legal_attacking_cards, self.legal_defending_cards)
        self.gui = GUI() if self.to_render else None
        self.deck = Deck()
        self.trump_rank = self.deck.HEARTS
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
        return self.state[0][:], self.state[1][:], self.state[2][:], self.state[3][:]

    def initialize_players(self):
        for player in self.active_players:
            player.set_gui(self.gui)
            player.initialize_for_game()
        deal = True
        while deal:
            self.initialize_deck()
            self.reset_hands()
            deal = self.deal_cards()

    def initialize_deck(self):
        self.deck = Deck()
        self.deck.shuffle()
        trump_card = self.deck.draw()[0]
        self.trump_rank = trump_card[1]
        self.deck.to_bottom(trump_card)

    def reset_hands(self):
        for player in self.active_players:
            player.empty_hand()
            player.set_trump_rank(self.trump_rank)

    def deal_cards(self):
        for player in self.active_players:
            drawn_cards = self.deck.draw(self.HAND_SIZE)
            player.take_cards(drawn_cards)
            if not player.is_starting_hand_legal():
                return True
        return False

    def set_first_attacker(self):
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

    def step(self, action) -> Tuple[StateType, NumberType, bool, InfoType]:
        self.last_action = action
        self.reward = 0
        if self.attack_phase:
            self.do_attack_phase()
        else:
            self.do_defence_phase()
        self.check_end_round()
        if self.reset_round:
            self.update_players_hands()
            self.remove_winners()
            self.update_active_players_order()
            if not self.game_over():
                self.do_reset_round()
        self.update_legal_cards()
        return (self.state[0][:], self.state[1][:], self.state[2][:], self.state[3][:]), self.reward, self.game_over(), list()

    def do_attack_phase(self):
        if self.last_action != self.deck.NO_CARD:
            self.attacking_cards.append(self.last_action)
            for player in self.active_players:
                player.update_round_progress(
                    self.turn_player.name, self.last_action, True)
            self.attack_phase = False
            if self.turn_player.hand_size == 0:
                self.reward = 30
        else:
            if self.turn_player == self.active_players[-1]:
                self.defending = False
                self.successful = True
                self.reset_attacker = True
            else:
                self.reset_attacker = False
                self.turn_player = self.active_players[self.active_players.index(
                    self.turn_player) + 1]
                if self.turn_player == self.active_players[self.defender]:
                    if self.turn_player != self.active_players[-1]:
                        self.turn_player = self.active_players[self.active_players.index(
                            self.turn_player) + 1]
                    else:
                        self.turn_player = self.active_players[self.active_players.index(
                            self.turn_player) - 1]
                        self.defending = False
                        self.successful = True
                        self.reset_attacker = True

    def do_defence_phase(self):
        if self.last_action != self.deck.NO_CARD:
            self.defending_cards.append(self.last_action)
            for player in self.active_players:
                player.update_round_progress(
                    self.active_players[self.defender].name, self.last_action, False)
            self.attack_phase = True
            if len(self.attacking_cards) == len(self.defending_cards) == self.limit:
                # successful defence
                self.defending = False
                self.successful = True
            self.reset_attacker = True
        else:
            self.defending = False
            self.successful = False
            self.turn_player.take_cards(self.attacking_cards)
            self.turn_player.take_cards(self.defending_cards)

    def update_players_hands(self):
        for player in self.active_players[:self.defender] + self.active_players[self.defender + 1:]:
            drawn_cards = self.deck.draw(
                max(0, self.HAND_SIZE - player.hand_size))
            player.take_cards(drawn_cards)
        drawn_cards = self.deck.draw(
            max(0, self.HAND_SIZE - self.active_players[self.defender].hand_size))
        self.active_players[self.defender].take_cards(drawn_cards)

    def remove_winners(self):
        to_remove = list()
        for player in self.active_players:
            if player.hand_size == 0:
                to_remove.append(player)
        for player in to_remove:
            self.active_players.remove(player)

    def update_active_players_order(self):
        if len(self.active_players) == 1:
            self.loser = self.active_players[0]
        elif len(self.active_players) >= self.MIN_PLAYERS:
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

    def do_reset_round(self):
        self.attacking_cards = list()
        self.defending_cards = list()
        self.legal_attacking_cards = list()
        self.legal_defending_cards = list()
        self.state = (self.attacking_cards, self.defending_cards,
                      self.legal_attacking_cards, self.legal_defending_cards)
        self.turn_player = self.active_players[self.attacker]
        self.attacking_player = self.active_players[self.attacker]
        self.defending = True
        self.successful = False
        self.limit = min(
            self.HAND_SIZE, self.active_players[self.defender].hand_size)
        self.attack_phase = True
        self.reset_attacker = True
        self.reset_round = False

    def update_legal_cards(self):
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
                                ((card[1] == self.trump_rank) and (self.attacking_cards[-1][1] != self.trump_rank)):
                            self.legal_defending_cards.append(card)
        self.state = (self.attacking_cards, self.defending_cards,
                      self.legal_attacking_cards, self.legal_defending_cards)

    def dispose_events(self):
        self.gui.dispose_events()

    def check_end_round(self):
        if not self.defending:
            if self.turn_player == self.active_players[self.defender]:
                if self.successful:
                    self.reward = len(self.defending_cards)
                else:
                    self.reward = -(len(self.attacking_cards) +
                                    len(self.defending_cards))
            self.reset_round = True
        else:
            if self.attack_phase and self.reset_attacker:
                self.turn_player = self.active_players[self.attacker]
            elif not self.attack_phase:
                self.turn_player = self.active_players[self.defender]

    def render(self):
        if self.to_render and self.gui is not None:
            attacker = self.active_players[self.attacker] if len(
                self.active_players) else None
            defender = self.active_players[self.defender] if len(
                self.active_players) >= self.MIN_PLAYERS else None
            self.gui.show_screen(self.players, (self.attacking_cards, self.defending_cards),
                                 attacker, defender,
                                 self.deck, self.trump_rank)

    def game_over(self):
        return len(self.active_players) < self.MIN_PLAYERS

    def get_turn_player(self) -> Optional[DurakPlayer]:
        return self.turn_player

    def to_attack(self) -> bool:
        return self.attack_phase

    def end_gui(self):
        self.gui.end() if self.gui is not None else None

    def get_loser(self):
        return self.loser

    


# proper running of a durak game from the environment:
num_games = 100
game = DurakEnv([NFSPPlayer(DurakEnv.HAND_SIZE, "NFSP Player"), HumanPlayer(
    DurakEnv.HAND_SIZE, "Human"), RandomPlayer(DurakEnv.HAND_SIZE, "Random Player")], False)
for game_index in range(num_games):
    state = game.reset()
    game.render()
    while True:
        turn_player = game.get_turn_player()
        to_attack = game.to_attack()
        act = turn_player.get_action(state, to_attack)
        new_state, reward, done, info = game.step(act)
        if isinstance(turn_player, LearningPlayer):
            turn_player.learn_step(state, new_state, reward, info)
        state = new_state
        game.render()
        if done:
            break
    loser = game.get_loser()
    if loser is not None:
        print(loser.name, "lost")
    else:
        print("ended in a tie")
print('done')
