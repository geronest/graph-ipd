"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of deep advantage network
interfaces based on openAI gym
"""

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent


class DANAgent(Agent):
    def __init__(
        self, env, config, state_manager, name="dan", seed_modifier=0, verbose=False
    ):
        super().__init__(env, config, state_manager)
        config_model = config[name]

        self.name = "dan"
        # fix num_actions to 1 when used for partner selection
        # TODO: might have to change.
        if name == "dan_ps":
            self.num_actions = 1
            self.name += "_ps"

        self.sanity_check(config_model)

        # define network
        torch.manual_seed(config.random_seeds.agent + seed_modifier)

        # self.layers = list()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            nn.Linear(
                self.dim_state, config_model.dims_hidden_layers[0], dtype=torch.float32
            ),
        )
        # hidden layers
        for idx_hl in range(len(config_model.dims_hidden_layers) - 1):
            self.layers.append(
                nn.Linear(
                    config_model.dims_hidden_layers[idx_hl],
                    config_model.dims_hidden_layers[idx_hl + 1],
                    dtype=torch.float32,
                ),
            )
        # V layer
        self.layers.append(
            nn.Linear(
                config_model.dims_hidden_layers[-1],
                1,
                dtype=torch.float32,
            ),
        )
        # A layer
        self.layers.append(
            nn.Linear(
                config_model.dims_hidden_layers[-1],
                self.num_actions,
                dtype=torch.float32,
            ),
        )

        self.layers_target = copy.deepcopy(self.layers)

        self.learning_rate = config_model.learning_rate
        self.epsilon = nn.Parameter(
            torch.tensor(config_model.epsilon, dtype=torch.float),
            requires_grad=False,
        )
        self.gamma = config_model.gamma

        self.optim = torch.optim.SGD(self.parameters(), self.learning_rate)
        self.step_count = 0
        self.freq_refresh_target = config_model.freq_refresh_target
        self.losses = deque(maxlen=config.record_step)

        if verbose:
            print(f"[Deep A-learning: {name}] initialised")

    def sanity_check(self, config):
        assert config.learning_rate >= 0.0
        assert (config.epsilon >= 0.0) and (config.epsilon <= 1.0)
        assert (config.gamma >= 0.0) and (config.gamma <= 1.0)
        assert isinstance(config.freq_refresh_target, int) and (
            config.freq_refresh_target > 0
        )

    def forward_dan(self, layers, x):
        """
        forward one of {training, target} networks.
        Arguments:
            layers
            x
        Returns:
            network output given x
        """
        for layer in layers[:-2]:
            x = F.relu(layer(x))
            # x = nn.ReLU()(layer(x))
        return layers[-2](x) + layers[-1](x)

    def forward(self, x):
        return self.forward_dan(self.layers, x)

    def add_name(self, name):
        """
        add a string to the agent's name.
        * made for partner selection module, pairing it with different types of agents
        """
        self.name += "_" + name
        pass

    def act(self, state):
        """
        given a state information, return action of the agent.
        Arguments:
            state
        Returns:
            action
        """
        state = torch.tensor(state, dtype=torch.float32)
        # apply epsilon
        if torch.rand((1)) < self.epsilon:  # random action
            action = int(torch.rand((1)) * self.num_actions)
        else:  # choose action with highest q value
            action = self(state).argmax().item()

        return action

    def step_train(self, state, action, reward, new_state):
        """
        single step of the training process.
        """
        state = torch.tensor(state, dtype=torch.float32)
        new_state = torch.tensor(new_state, dtype=torch.float32)

        self.optim.zero_grad()
        oldq = self(state)[action]
        newq = (
            reward
            + self.gamma
            * self.forward_dan(self.layers_target, new_state).max().detach()
        )
        loss = (newq - oldq) ** 2  # TODO: Exponential Moving Average?
        self.losses.append(loss.item())

        loss.backward()
        self.optim.step()

        # update target network
        self.step_count += 1
        if self.step_count % self.freq_refresh_target == 0:
            for idx_layer in range(len(self.layers_target)):
                self.layers_target[idx_layer].load_state_dict(
                    self.layers[idx_layer].state_dict()
                )

        if self.config.print_loss and (self.step_count % self.config.record_step) == 0:
            avg_loss = np.mean(self.losses)
            print(
                "[DANAgent-step_train()] average loss across"
                + f"recent {self.config.record_step} steps: "
                + f"{avg_loss:.4f}"
            )
            print(f"[DANAgent-step_train()] Q-values: {self(state)}")

    def train(self):
        """
        run training.
        """
        pass

    def save(self, path):
        """
        save model parameters at the designated path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        load model parameters from the designated path.
        """
        self.load_state_dict(torch.load(path))
