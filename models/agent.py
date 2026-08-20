"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

define basic structure of Agent object.
"""

from collections import deque

import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, idx_agent, env, config, state_manager):
        super().__init__()

        self.config = config
        if config.device == "gpu":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.idx_me = idx_agent
        self.name = f"{idx_agent}_agent"
        self.num_actions = env.num_actions
        self.dim_state = state_manager.dim_state

        self.stm = state_manager
        self.selected_history = config.selected_history

        self.history = dict()
        for k in ["states", "actions", "rewards", "nstates", "ids_op"]:
            self.history[k] = deque(maxlen=config.state.len_history)

    def getattr(self, attr):
        return getattr(self, attr)

    def get_extra_state(self):
        return ""

    def set_extra_state(self, v):
        pass

    def get_name(self):
        return self.name

    def produce_state(self):
        return self.stm.encode(self.history, self.idx_me)

    def act(self, state):
        """
        given a state information, return action of the agent.
        inputs:
            state
        return:
            action
        """
        pass

    def store(self, k, v):
        """
        store history of actions and states.
        inputs:
            k: name of the value that is to be stored
            v: value
        """
        self.history[k].append(v)

    def store_multiple(self, d_items):
        for k in d_items:
            self.store(k, d_items[k])

    def step_train(self, state, action, reward, new_state):
        """
        single step of the training process.
        """
        pass

    def train(self):
        """
        run training.
        """
        pass

    def save(self, path):
        """
        save trained model in a given directory.
        """
        pass

    def load(self, path):
        """
        load a trained model from a given directory.
        """
        pass

    def rewire(self, neighbours, states_op):
        return {"idx_me": self.idx_me, "attach": None, "detach": None}
