"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

define environments for game theory experiments.
interfaces based on openAI gym
"""

import numpy as np


class MatrixGameEnv:
    def __init__(self, config):
        """
        set matrix for game with information from config.
        (num_players, num_actions ** num_players)
        """
        self.num_players = config.num_players
        self.num_actions = config.num_actions
        self.multiplier = config.multiplier
        self.mat = np.zeros((self.num_players, self.num_actions**self.num_players))

        # assign score for each action pair
        for idx_player in range(self.num_players):
            self.mat[idx_player] = np.array(config.scores[idx_player]) * self.multiplier

    def reset(self):
        """
        reset the environment.

        return:
            new_state: numpy array of players' states
        """
        new_states = np.zeros(self.num_players, dtype=np.int32)
        return new_states

    def get_idx_actions(self, actions):
        """
        retrieve idx for actions in the matrix.
        """
        idx = 0
        for action in actions:
            idx = idx * self.num_actions + action
        return idx

    def step(self, actions):
        """
        retrieving actions from players, return score for each.

        inputs:
            action: tuple of players' actions

        return:
            rewards: numpy array of players' rewards
            new_states: numpy array of next step's states for each player.
        """
        rewards = np.zeros(self.num_players)
        idx_actions = self.get_idx_actions(actions)

        for idx_player in range(self.num_players):
            rewards[idx_player] = self.mat[idx_player][idx_actions]

        # NOTE: currently not used
        new_states = np.zeros(self.num_players, dtype=np.int32)

        return rewards, new_states
