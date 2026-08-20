"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

define constraints for prisoner's dilemma.
interfaces based on openAI gym
"""

from .matgame import MatrixGameEnv


class PrisonersDilemmaEnv(MatrixGameEnv):
    def __init__(self, config):
        """
        additional constraint check on score matrix
        """
        super().__init__(config)

        assert self.num_players == 2
        assert self.num_actions == 2
        assert self.mat.shape == (2, 4)

        # TODO: constraint check on self.mat
