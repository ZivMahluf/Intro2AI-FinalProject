import torch
import torch.nn as nn
import numpy as np

import random


NUM_ACTIONS = 37
INPUT_SIZE = 37 + 36*4
INPUT_SHAPE = (INPUT_SIZE,)
HIDDEN_LAYER_DIM = 37  # todo maybe 32, 64


def DQN():
    return DQNBase()


class DQNBase(nn.Module):
    """
    Basic DQN

    parameters
    ---------
    env         environment(openai gym)
    """

    def __init__(self):
        super(DQNBase, self).__init__()

        self.input_shape = INPUT_SHAPE
        self.num_actions = NUM_ACTIONS  # todo - make constant somewhere
        self.hidden_layer_dim = HIDDEN_LAYER_DIM

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
            nn.Linear(64, self.num_actions),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

    def act(self, state, epsilon, legal_cards):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_value = self.forward(state).cpu().detach().numpy().flatten()
                q_value[np.array(legal_cards) == 0] = -1 * float('inf')
                action = np.random.choice(
                    np.array([np.argmax(q_value)]).flatten())
        else:
            action = np.random.choice(
                np.array([np.nonzero(legal_cards)]).flatten())
        return action


class Policy(DQNBase):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """

    def __init__(self):
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
            nn.Linear(64, self.num_actions),
            nn.Softmax(dim=1),
        )

    def act(self, state, epsilon, legal_cards):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            new_state = state.unsqueeze(0)
            distribution = self.forward(
                new_state).cpu().detach().numpy().flatten()
            distribution[np.array(legal_cards) == 0] = -1 * float('inf')
            action = np.random.choice(
                np.array([np.argmax(distribution)]).flatten())
            return action
