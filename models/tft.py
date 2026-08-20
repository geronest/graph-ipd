"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of tit-for-tat
interfaces based on openAI gym
"""

import torch

from .agent import Agent


class TitForTatAgent(Agent):
    def __init__(
        self, env, config, state_manager, name="tft", seed_modifier=0, verbose=False
    ):
        super().__init__(env, config, state_manager)

        self.sanity_check(config)
        self.name = "tft"
        if self.name not in config or "epsilon" not in config[self.name]:
            self.epsilon = 0.0
        else:
            self.epsilon = config[self.name]["epsilon"]
        self.idx_last_history = state_manager.idx_last_action

        torch.manual_seed(config.random_seeds.agent + seed_modifier)

        if verbose:
            print("[Tit-for-Tat: Agent] Initialised")

    def sanity_check(self, config):
        pass

    def act(self, state, prepare_train=None):
        """
        returns action index.

        inputs:
            state: integer of *opponent's action history

        returns:
            action: integer
        """

        # apply epsilon
        if torch.rand((1)) < self.epsilon:  # random action
            action = int(torch.rand((1)) * self.num_actions)
        else:
            # choose action with the same index of state.
            # -> the same action the opponent selected *last time*
            action = state[
                self.idx_last_history : self.idx_last_history + self.num_actions
            ].argmax()
            # print(f"[TFT] opponent's action is {state}. Selected {action} for action")
        return action
