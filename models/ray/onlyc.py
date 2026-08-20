"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of only cooperating agent
interfaces based on openAI gym
"""

import ray
import torch

from ..agent import Agent


@ray.remote(num_gpus=0.0005, num_cpus=0.01)
class RayOnlyCooperateAgent(Agent):
    def __init__(
        self,
        idx_agent,
        env,
        config,
        state_manager,
        name="onlyc",
        seed_modifier=0,
        verbose=False,
    ):
        self.sanity_check(config)
        super().__init__(idx_agent, env, config, state_manager)

        self.name = "onlyc"
        if self.name not in config or "epsilon" not in config[self.name]:
            self.epsilon = 0.0
        else:
            self.epsilon = config[self.name]["epsilon"]

        torch.manual_seed(config.random_seeds.agent + seed_modifier)

        if verbose:
            print("[OnlyCooperate: Agent] Initialised")

    def sanity_check(self, config):
        pass

    def act(self, state, prepare_train=None):
        """
        returns cooperating action.

        inputs:
            state: integer of *opponent's action history

        returns:
            action: integer
        """

        # apply epsilon
        if torch.rand((1)) < self.epsilon:  # random action
            action = int(torch.rand((1)) * self.num_actions)
        else:
            action = 1

        return action

    def step_train(self, new_state):
        pass

    def reset_buffer(self):
        pass

    def store_buffer(self, key, value):
        pass
