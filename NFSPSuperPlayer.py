from NFSPPlayer import NFSPPlayer


class NFSPSuperPlayer(NFSPPlayer):
    def __init__(self, hand_size, name, device='cpu'):
        super().__init__(hand_size,name,device)
        self.attackPlayer = NFSPPlayer(hand_size, name, device)
        self.defendPlayer = NFSPPlayer(hand_size, name, device)
        self.attacked = False

    def attack(self, table, legal_cards_to_play):
        self.attacked = True
        self.attackPlayer._hand = self._hand
        act = self.attackPlayer.attack(table, legal_cards_to_play)
        self._hand = self.attackPlayer._hand
        return act

    def defend(self, table, legal_cards_to_play):
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

    def learn_step(self, old_state, new_state, action, reward):
        if self.attacked:
            self.attackPlayer.learn_step(
                old_state, new_state, action, reward)
        else:
            self.defendPlayer.learn_step(
                old_state, new_state, action, reward)
