"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of deep q network
interfaces based on openAI gym
"""

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent


def calculate_eps(config, current_step=0):
    """
    apply epsilon decay when specified.
    """
    if config.decay_epsilon:
        coef_eps = 1 - (current_step / config.num_steps) * (
            1 / config.decay_epsilon_schedule
        )
        coef_eps = max(coef_eps, 0.0)
    else:
        coef_eps = 1

    return coef_eps


class DQNAgent(Agent):
    def __init__(
        self,
        idx_agent,
        env,
        config,
        state_manager,
        name="dqn",
        seed_modifier=0,
        verbose=False,
    ):
        super().__init__(idx_agent, env, config, state_manager)
        config_model = config[name]

        self.name = "dqn"
        # fix num_actions to 1 when used for partner selection
        # TODO: might have to change.
        if name == "dqn_ps":
            self.num_actions = 1
            self.name += "_ps"

        self.sanity_check(config_model)

        # define network
        torch.manual_seed(config.random_seeds.agent + seed_modifier)

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            nn.Linear(
                self.dim_state,
                config_model.dims_hidden_layers[0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # hidden layers
        for idx_hl in range(len(config_model.dims_hidden_layers) - 1):
            self.layers.append(
                nn.Linear(
                    config_model.dims_hidden_layers[idx_hl],
                    config_model.dims_hidden_layers[idx_hl + 1],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        # output layer
        self.layers.append(
            nn.Linear(
                config_model.dims_hidden_layers[-1],
                self.num_actions,
                dtype=torch.float32,
                device=self.device,
            ),
        )

        self.layers_target = copy.deepcopy(self.layers)

        self.learning_rate = config_model.learning_rate
        self.epsilon = nn.Parameter(
            torch.tensor(config_model.epsilon, dtype=torch.float, device=self.device),
            requires_grad=False,
        )
        self.gamma = config_model.gamma

        # self.optim = torch.optim.SGD(
        self.optim = torch.optim.Adam(
            self.parameters(),
            self.learning_rate,
            weight_decay=config_model.weight_decay,
        )
        self.step_count = 0
        self.freq_refresh_target = config_model.freq_refresh_target
        self.losses = deque(maxlen=config.record_step)
        self.cached_Q = None

        if verbose:
            print(f"[Deep Q-learning: {name}] initialised")

    def sanity_check(self, config):
        assert config.learning_rate >= 0.0
        assert (config.epsilon >= 0.0) and (config.epsilon <= 1.0)
        assert (config.gamma >= 0.0) and (config.gamma <= 1.0)
        assert isinstance(config.freq_refresh_target, int) and (
            config.freq_refresh_target > 0
        )

    def forward_dqn(self, layers, x):
        """
        forward one of {training, target} networks.
        Arguments:
            layers
            x
        Returns:
            network output given x
        """
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        for layer in layers[:-1]:
            x = F.relu(layer(x))
        out = layers[-1](x)
        if out.isnan().any():
            print(f"[{self.name}-Forward_DQN] isNaN")
            raise ValueError
        return out
        # return layers[-1](x)

    def forward(self, x):
        return self.forward_dqn(self.layers, x)

    def add_name(self, name):
        """
        add a string to the agent's name.
        * made for partner selection module, pairing it with different types of agents
        """
        self.name += "_" + name
        pass

    def act(self, state, prepare_train=False):
        """
        given a state information, return action of the agent.
        Arguments:
            state
        Returns:
            action
        """
        if prepare_train:
            self.cached_Q = self(state)
        with torch.no_grad():
            # apply epsilon
            if (
                torch.rand((1), device=torch.device(self.device)) < self.epsilon
            ):  # random action
                action = int(torch.rand((1)) * self.num_actions)
            else:  # choose action with highest q value
                if prepare_train:
                    action = self.cached_Q.argmax().item()
                else:
                    action = self(state).argmax().item()

        return action

    def step_train(self, state, action, reward, new_state):
        """
        single step of the training process.
        """
        self.optim.zero_grad(set_to_none=True)

        # oldq = self.cached_Q[action]
        oldq = self(state)[action]
        with torch.no_grad():
            newq = (
                reward
                + self.gamma * self.forward_dqn(self.layers_target, new_state).max()
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
                "[DQNAgent-step_train()] average loss across"
                + f"recent {self.config.record_step} steps: "
                + f"{avg_loss:.4f}"
            )
            print(f"[DQNAgent-step_train()] Q-values: {self(state)}")

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
