from Types import List
from Deck import Deck
import torch
import torch.nn as nn
import numpy as np
import random


NUM_ACTIONS = len(Deck.get_full_list_of_cards()) + 1
INPUT_SIZE = 1 + 5 * len(Deck.get_full_list_of_cards())
INPUT_SHAPE = (INPUT_SIZE,)


class DQNBase(nn.Module):

    def __init__(self):
        """
        Constructor.
        Defines the neural network that will be used.
        """
        super(DQNBase, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: Network input - vector of length INPUT_SIZE.
        :return: Network output.
        """
        x = self.fc(x)
        return x

    def act(self, state: torch.Tensor, epsilon: float, legal_cards: List[int]) -> int:
        """
        Decides on an action to take based on the given state, epsilon, and legal cards.
        :param state: A torch Tensor representing a state and holding the network input.
        :param epsilon: Epsilon for epsilon-greedy acting.
        :param legal_cards: List of indicators of the legal cards to play.
        :return: Index of an action in range [0, NUM_ACTIONS)
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_values = self.forward(state).cpu().detach().numpy().flatten()
                q_values[np.array(legal_cards) == 0] = -1 * float('inf')
                action = np.random.choice(np.array([np.argmax(q_values)]).flatten())
        else:
            action = np.random.choice(np.array([np.nonzero(legal_cards)]).flatten())
        return action


class Policy(DQNBase):

    def __init__(self):
        """
        Constructor.
        Defines the policy's neural network.
        """
        super(Policy, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
            nn.Softmax(dim=1),
        )

    def act(self, state, epsilon, legal_cards):
        """
        Decides on an action to take based on the given state and legal cards.
        :param state: A torch Tensor representing a state and holding the network input.
        :param epsilon: Epsilon for epsilon-greedy acting (for this specific method, ignored).
        :param legal_cards: List of indicators of the legal cards to play.
        :return: Index of an action in range [0, NUM_ACTIONS)
        """
        with torch.no_grad():
            new_state = state.unsqueeze(0)
            distribution = self.forward(new_state).cpu().detach().numpy().flatten()
            distribution[np.array(legal_cards) == 0] = -np.inf
            action = np.random.choice(np.array([np.argmax(distribution)]).flatten())
            return action
