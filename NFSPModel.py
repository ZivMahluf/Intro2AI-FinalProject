import torch
import torch.nn as nn
import torch.nn.functional as F

import random


NUM_ACTIONS = 37
INPUT_SHAPE = (37,)
INPUT_SIZE = 37


def DQN(is_dueling):
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

        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action


class Policy(DQNBase):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """

    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return action
