from Types import NumberType, CardType, CardListType, TableType, StateType
from NFSPPlayer import NFSPPlayer


class NFSPSuperPlayer(NFSPPlayer):
    def __init__(self, hand_size: int, name: str, device: str = 'cpu'):
        super().__init__(hand_size, name, device)
        self.attackPlayer = NFSPPlayer(hand_size, name, device)
        self.defendPlayer = NFSPPlayer(hand_size, name, device)
        self.attacked = False

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        self.attacked = True
        self.attackPlayer._hand = self._hand
        act = self.attackPlayer.attack(table, legal_cards_to_play)
        self._hand = self.attackPlayer._hand
        return act

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        self.attacked = False
        self.defendPlayer._hand = self._hand
        act =  self.defendPlayer.defend(table, legal_cards_to_play)
        self._hand = self.defendPlayer._hand
        return act

    def save_network(self, name: str):
        """
        saves the neural network for future use
        Parameters
        ----------
        name: name of the file that will store the neural network
        """
        self.attackPlayer.save_network(name+"attack")
        self.defendPlayer.save_network(name+"defend")

    def learn_step(self, old_state: StateType, new_state: StateType, action: CardType, reward: NumberType, old_hand) -> None:
        if self.attacked:
            self.attackPlayer.learn_step(old_state, new_state, action, reward, old_hand)
        else:
            self.defendPlayer.learn_step(old_state, new_state, action, reward, old_hand)
