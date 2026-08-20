"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of tabular q learning
interfaces based on openAI gym
"""

import torch
import torch.nn as nn

from .agent import Agent


class TabularQLearningAgent(Agent):
    def __init__(
        self,
        idx_agent,
        env,
        config,
        state_manager,
        name="tabularq",
        seed_modifier=0,
        verbose=False,
    ):
        super().__init__(idx_agent, env, config, state_manager)
        config_tql = config[name]

        self.name = "tabq"
        # fix num_actions to 1 when used for partner selection
        if name == "tabularq_ps":
            self.num_actions = 1
            self.name += "_ps"

        # setting random seed
        torch.manual_seed(config.random_seeds.agent + seed_modifier)

        self.config_state = config["state"]
        self.num_states = state_manager.num_states

        if config_tql.init_params == "random":
            init_qvalues = torch.rand(
                (self.num_states, self.num_actions), dtype=torch.float
            )
        elif config_tql.init_params == "zero":
            init_qvalues = torch.zeros(
                (self.num_states, self.num_actions), dtype=torch.float
            )
        self.qvalues = nn.Parameter(init_qvalues, requires_grad=False)
        self.sanity_check(config_tql)
        self.learning_rate = config_tql.learning_rate
        self.epsilon = nn.Parameter(
            torch.tensor(config_tql.epsilon, dtype=torch.float),
            requires_grad=False,
        )
        self.gamma = config_tql.gamma
        if verbose:
            print(
                f"[Tabular Q-learning: {name}] "
                + f"initialised with Q-values: {self.qvalues.data}"
            )

    def sanity_check(self, config):
        assert config.learning_rate >= 0.0
        assert (config.epsilon >= 0.0) and (config.epsilon <= 1.0)
        assert (config.gamma >= 0.0) and (config.gamma <= 1.0)

    def add_name(self, name):
        """
        add a string to the agent's name.
        * made for partner selection module, pairing it with different types of agents
        """
        self.name += "_" + name

    def convert_state_to_int(self, state):
        """
        convert vector-based state into an integer index for Q values
        * assumes each value in the vector is either 0 or 1
        """
        res = 0
        idx_state = 0
        for i in range(self.config_state.len_history):
            res = 2 * res + state[idx_state].argmax()
            if self.config_state.history_type == "pair":
                idx_state += 2
                res = 2 * res + state[idx_state].argmax()
        # TODO: opponent id processing. move to state manager later.

        return int(res)

    def act(self, state):
        """
        returns action index.

        inputs:
            state: integer

        returns:
            action: integer
        """
        # apply epsilon
        if torch.rand((1)) < self.epsilon:  # random action
            action = int(torch.rand((1)) * self.num_actions)
        else:  # choose action with highest q value
            state = self.convert_state_to_int(state)
            action = self.qvalues[state].argmax().item()

        return action

    def step_train(self, state, action, reward, new_state):
        """
        single step of the training process.
        """
        state = self.convert_state_to_int(state)
        new_state = self.convert_state_to_int(new_state)

        newq = (1 - self.learning_rate) * self.qvalues[
            state, action
        ] + self.learning_rate * (reward + self.gamma * self.qvalues[new_state].max())
        self.qvalues[state, action].fill_(newq)

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
