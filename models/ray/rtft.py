"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of tit-for-tat
interfaces based on openAI gym
"""

import ray
import torch

from ..agent import Agent


@ray.remote(num_gpus=0.0005, num_cpus=0.01)
class RayReverseTitForTatAgent(Agent):
    def __init__(
        self,
        idx_agent,
        env,
        config,
        state_manager,
        name="rtft",
        seed_modifier=0,
        verbose=False,
    ):
        super().__init__(idx_agent, env, config, state_manager)

        self.sanity_check(config)
        self.name = "rtft"
        if self.name not in config or "epsilon" not in config[self.name]:
            self.epsilon = 0.0
        else:
            self.epsilon = config[self.name]["epsilon"]
        self.idx_last_history = state_manager.idx_last_action

        torch.manual_seed(config.random_seeds.agent + seed_modifier)

        if verbose:
            print("[Reverse Tit-for-Tat: Agent] Initialised")

    def sanity_check(self, config):
        pass

    def act(self, state, prepare_train=None):
        """
        returns action index.
        * assumes there are only two actions available: 0 defect, 1 cooperate.

        inputs:
            state: integer of *opponent's action history

        returns:
            action: integer
        """

        # apply epsilon
        if torch.rand((1)) < self.epsilon:  # random action
            action = int(torch.rand((1)) * self.num_actions)
        else:
            # -> choose the OPPOSITE action the opponent selected *last time*
            action = (
                1
                - state[
                    self.idx_last_history : self.idx_last_history + self.num_actions
                ].argmax()
            )
            # print(f"[ReverseTFT] opponent's action is {state}."
            #       +f" Selected {action} for action")
        return action

    def step_train(self, new_state):
        pass

    def reset_buffer(self):
        pass

    def store_buffer(self, key, value):
        pass
