from NFSPPlayer import NFSPPlayer


class NFSPSuperPlayer(NFSPPlayer):
    def __init__(self, hand_size, name, device='cpu'):
        self.attackPlayer = NFSPPlayer(hand_size, name, device)
        self.defendPlayer = NFSPPlayer(hand_size, name, device)
        self.attacked = False

    def attack(self, table, legal_cards_to_play):
        self.attacked = True
        return self.attackPlayer.attack(table, legal_cards_to_play)

    def defend(self, table, legal_cards_to_play):
        self.attacked = False
        return self.defendPlayer.defend(table, legal_cards_to_play)

    def learn_step(self, old_state, new_state, action, reward, info):
        if self.attacked:
            self.attackPlayer.learn_step(
                old_state, new_state, action, reward, info)
        else:
            self.defendPlayer.learn_step(
                old_state, new_state, action, reward, info)
