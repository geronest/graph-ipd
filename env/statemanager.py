"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

functions for shaping state
"""

import numpy as np


class StateManager:
    def __init__(self, config, seed_modifier=0):

        self.sanity_check(config)
        self.rng = np.random.default_rng(seed=config.random_seeds.state + seed_modifier)

        self.history_type = config.state.history_type
        self.len_history = config.state.len_history
        self.agent_id = config.state.agent_id
        self.include_opponent_history = config.state.include_opponent_history
        self.num_actions = config.matgame.num_actions

        # calculate the number of agents
        self.num_agent = 0
        for k in config.population:
            self.num_agent += config.population[k]

        # calculate state dimension
        self.dim_state = 0
        self.num_states = self.num_actions**self.len_history
        if self.history_type == "single":
            self.dim_state += self.num_actions * self.len_history
            self.idx_last_action = self.dim_state - self.num_actions
        elif self.history_type == "pair":
            self.dim_state += (self.num_actions * 2) * self.len_history
            self.num_states = self.num_states**2
            self.idx_last_action = self.dim_state - (self.num_actions * 2)

        self.dim_id = 0
        self.idx_last_id = -1
        if self.agent_id == "onehot":
            self.dim_id = self.num_agent
        elif self.agent_id == "binary":
            self.dim_id = int(np.ceil(np.log2(self.num_agent)))

        if self.agent_id != "none":
            self.dim_state += self.dim_id
            self.num_states *= self.num_agent

            if self.include_opponent_history:
                self.dim_state += self.dim_id * self.len_history
                self.num_states *= self.num_agent**self.len_history
                self.idx_last_id = self.dim_state - (2 * self.dim_id)

        if config.profile_qvalues:
            if self.num_states > 1024:
                print(
                    "[StateManager-WARNING]"
                    + f"num_states: {self.num_states}. are you sure? "
                )
            self.possible_states = self.get_all_states()
            self.possible_state_ids = self.get_all_state_ids(self.possible_states)

    def sanity_check(self, config):
        """
        check the validity of configurations.
        """

        len_history = config.state.len_history
        if not isinstance(len_history, int) or len_history < 1:
            print(f"[ERROR-StateManager-sanity check]: len_history {len_history}")
            raise ValueError

        if config.state.agent_id not in ["none", "onehot", "binary"]:
            print(
                f"[ERROR-StateManager-sanity check]: agent_id {config.state.agent_id}"
            )
            raise ValueError

        if config.state.history_type not in ["single", "pair"]:
            print(
                "[ERROR-StateManager-sanity check]: history_type "
                + config.state.history_type
            )
            raise ValueError

    def encode(self, history, id_me=0):
        """
        encode the state vector.
        binary numbers: decimal 6 -> binary 110
        """

        state = np.zeros(self.dim_state)

        # 1: action history
        idx_state = 0

        for actions in history["actions"]:
            state[idx_state + actions[0]] = 1
            idx_state += self.num_actions

            if self.history_type == "pair":
                state[idx_state + actions[1]] = 1
                idx_state += self.num_actions

        # 2: agent identity
        # 2-1: opponent history
        if self.include_opponent_history:
            for id_op in history["ids_op"]:
                if self.agent_id == "binary":
                    for i in range(self.dim_id):
                        state[idx_state + (self.dim_id - i - 1)] = id_op % 2
                        id_op = id_op // 2
                    idx_state += self.dim_id
                elif self.agent_id == "onehot":
                    state[idx_state + id_op] = 1
                    idx_state += self.dim_id

        # 2-2: id_me
        if self.agent_id == "binary":
            for i in range(self.dim_id):
                state[idx_state + (self.dim_id - i - 1)] = id_me % 2
                id_me = id_me // 2
            idx_state += self.dim_id
        elif self.agent_id == "onehot":
            state[idx_state + id_me] = 1
            idx_state += self.dim_id

        assert idx_state == self.dim_state

        return state

    def get_last_history(self, state):
        """
        retrieve the last action history from the given state.
        arguments:
            state
        returns:
            last action history.
              shape (2,) if self.history_type == "single"
              shape (4,) if self.history_type == "pair"
        """

        if self.history_type == "single":
            size_history = self.num_actions
        elif self.history_type == "pair":
            size_history = self.num_actions * 2

        return state[self.idx_last_action : self.idx_last_action + size_history]

    def bin_to_dec(self, vec):
        res = 0
        for idx in range(vec.shape[0]):
            res += vec[self.dim_id - (idx + 1)] * (2**idx)
        return int(res)

    def decode_state_to_str(self, state):
        """
        decode state to string.
        * assumes binary action
        arguments:
            state
        returns:
            string
        """
        res = ""

        idx_action = 0
        idx_id = self.num_actions * self.len_history
        if self.history_type == "pair":
            idx_id *= idx_id

        for i in range(self.len_history):
            if state[idx_action] == 1:
                res += "D"
            else:
                res += "C"
            idx_action += 2
            if self.history_type == "pair":
                if state[idx_action] == 1:
                    res += "D"
                else:
                    res += "C"
                idx_action += 2

            if self.include_opponent_history:
                if self.agent_id == "binary":
                    res += str(self.bin_to_dec(state[idx_id : idx_id + self.dim_id]))
                elif self.agent_id == "onehot":
                    res += str(state[idx_id : idx_id + self.dim_id].argmax())
                idx_id += self.dim_id
            res += "_"

        if self.agent_id != "none":
            if self.agent_id == "binary":
                res += str(self.bin_to_dec(state[idx_id : idx_id + self.dim_id]))
            elif self.agent_id == "onehot":
                res += str(state[idx_id : idx_id + self.dim_id].argmax())
            idx_id += self.dim_id
        else:
            res = res[:-1]

        return res

    def get_all_state_ids(self, states):
        """
        get all the possible combinations possible in the setting.

        returns:
            state_ids
        """
        res = list()
        for i in range(states.shape[0]):
            res.append(self.decode_state_to_str(states[i]))
        return res

    def produce_combination(self, m1, m2):
        """
        produce combination of the given matrix.
        arguments:
            m1: shape (a, b)
            m2: shape (c, d)
        returns:
            matrix of shape (a*c, b+d)
        """
        return np.concatenate(
            [np.repeat(m1, m2.shape[0], axis=0), np.tile(m2, (m1.shape[0], 1))], axis=1
        )

    def get_all_states(self):
        """
        get all the possible combinations possible in the setting.

        returns:
            states
        """
        # construct actual states
        # res = np.zeros((self.num_states, self.dim_state))

        actions = np.eye(self.num_actions)
        if self.history_type == "pair":
            actions = self.produce_combination(actions, actions)

        # 1: action history
        res = actions
        for i in range(self.len_history - 1):
            res = self.produce_combination(res, actions)

        # 2: agent identity
        # 2-1: opponent history
        if self.agent_id == "onehot":
            agent_ids = np.eye(self.num_agent)
        elif self.agent_id == "binary":
            binary_ids = np.arange(2).reshape((2, 1))
            agent_ids = binary_ids
            for i in range(self.dim_id - 1):
                agent_ids = self.produce_combination(agent_ids, binary_ids)

        if self.include_opponent_history:
            for i in range(self.len_history):
                res = self.produce_combination(res, agent_ids)

        # 2-2: id_me
        if self.agent_id != "none":
            res = self.produce_combination(res, agent_ids)

        return res
