from collections import deque

import numpy as np
from omegaconf import OmegaConf

from env.statemanager import StateManager
from utils.configs import init_config


def test_statemanager():

    # history: single-len1, agent_id: none
    config = OmegaConf.load("./configs/test/states_single_len1_none.yaml")
    config = init_config(config)
    stm = StateManager(init_config(config))

    histories = list()
    for i in range(4):
        histories.append(
            {
                "actions": deque(maxlen=config.state.len_history),
                "ids_op": deque(maxlen=config.state.len_history),
            }
        )
    histories[0]["actions"].append([1, 1])  # agent 0
    histories[1]["actions"].append([1, 0])  # agent 1
    histories[2]["actions"].append([0, 1])  # agent 2
    histories[3]["actions"].append([0, 1])  # agent 3
    assert (stm.encode(histories[0]) == np.array([0, 1])).all()
    assert (stm.encode(histories[1]) == np.array([0, 1])).all()
    assert (stm.encode(histories[2]) == np.array([1, 0])).all()
    assert (stm.encode(histories[3]) == np.array([1, 0])).all()
    assert (stm.possible_states == np.array([[1, 0], [0, 1]])).all()
    assert (stm.possible_state_ids == np.array(["D", "C"])).all()

    # history: single-len2, agent_id: none
    config = OmegaConf.load("./configs/test/states_single_len2_none.yaml")
    config = init_config(config)
    stm = StateManager(init_config(config))
    histories = list()
    for i in range(4):
        histories.append(
            {
                "actions": deque(maxlen=config.state.len_history),
                "ids_op": deque(maxlen=config.state.len_history),
            }
        )
    histories[0]["actions"].append([0, 1])  # agent 0
    histories[1]["actions"].append([1, 0])  # agent 1
    histories[2]["actions"].append([0, 0])  # agent 2
    histories[3]["actions"].append([1, 1])  # agent 3
    histories[0]["actions"].append([1, 1])  # agent 0
    histories[1]["actions"].append([1, 0])  # agent 1
    histories[2]["actions"].append([0, 1])  # agent 2
    histories[3]["actions"].append([0, 1])  # agent 3

    assert (stm.encode(histories[0]) == np.array([1, 0, 0, 1])).all()
    assert (stm.encode(histories[1]) == np.array([0, 1, 0, 1])).all()
    assert (stm.encode(histories[2]) == np.array([1, 0, 1, 0])).all()
    assert (stm.encode(histories[3]) == np.array([0, 1, 1, 0])).all()
    assert (
        stm.possible_states
        == np.array(
            [
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
            ]
        )
    ).all()
    assert (stm.possible_state_ids == np.array(["D_D", "D_C", "C_D", "C_C"])).all()

    # history: single-len2, agent_id: onehot, include_opponent_history: True
    histories[0]["ids_op"].append(3)
    histories[1]["ids_op"].append(0)
    histories[2]["ids_op"].append(1)
    histories[3]["ids_op"].append(0)
    histories[0]["ids_op"].append(2)
    histories[1]["ids_op"].append(2)
    histories[2]["ids_op"].append(0)
    histories[3]["ids_op"].append(1)

    config = OmegaConf.load("./configs/test/states_single_len2_onehot_ophis.yaml")
    config = init_config(config)
    stm = StateManager(init_config(config))
    assert (
        stm.encode(histories[0], 0)
        == np.array([1, 0, 0, 1] + [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    ).all()
    assert (
        stm.encode(histories[1], 1)
        == np.array([0, 1, 0, 1] + [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    ).all()
    assert (
        stm.encode(histories[2], 2)
        == np.array([1, 0, 1, 0] + [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    ).all()
    assert (
        stm.encode(histories[3], 3)
        == np.array([0, 1, 1, 0] + [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    ).all()

    # history: single-len2, agent_id: binary, include_opponent_history: False
    config = OmegaConf.load("./configs/test/states_single_len2_binary.yaml")
    config = init_config(config)
    stm = StateManager(init_config(config))
    assert (stm.encode(histories[0], 0) == np.array([1, 0, 0, 1] + [0, 0])).all()
    assert (stm.encode(histories[1], 1) == np.array([0, 1, 0, 1] + [0, 1])).all()
    assert (stm.encode(histories[2], 2) == np.array([1, 0, 1, 0] + [1, 0])).all()
    assert (stm.encode(histories[3], 3) == np.array([0, 1, 1, 0] + [1, 1])).all()
    assert (
        stm.possible_states
        == np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
                [1, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 1],
                [1, 0, 0, 1, 1, 0],
                [1, 0, 0, 1, 1, 1],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 1],
                [0, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 1, 1],
                [0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 1, 0],
                [0, 1, 0, 1, 1, 1],
            ]
        )
    ).all()
    assert (
        stm.possible_state_ids
        == np.array(
            [
                "D_D_0",
                "D_D_1",
                "D_D_2",
                "D_D_3",
                "D_C_0",
                "D_C_1",
                "D_C_2",
                "D_C_3",
                "C_D_0",
                "C_D_1",
                "C_D_2",
                "C_D_3",
                "C_C_0",
                "C_C_1",
                "C_C_2",
                "C_C_3",
            ]
        )
    ).all()

    # history: single-len2, agent_id: binary, include_opponent_history: True
    config = OmegaConf.load("./configs/test/states_single_len2_binary_ophis.yaml")
    config = init_config(config)
    stm = StateManager(init_config(config))
    assert (
        stm.encode(histories[0], 0) == np.array([1, 0, 0, 1] + [1, 1, 1, 0, 0, 0])
    ).all()
    assert (
        stm.encode(histories[1], 1) == np.array([0, 1, 0, 1] + [0, 0, 1, 0, 0, 1])
    ).all()
    assert (
        stm.encode(histories[2], 2) == np.array([1, 0, 1, 0] + [0, 1, 0, 0, 1, 0])
    ).all()
    assert (
        stm.encode(histories[3], 3) == np.array([0, 1, 1, 0] + [0, 0, 0, 1, 1, 1])
    ).all()

    # history: pair-len2, agent_id: none

    config = OmegaConf.load("./configs/test/states_pair_len2_none.yaml")
    config = init_config(config)
    stm = StateManager(init_config(config))
    assert (stm.encode(histories[0]) == np.array([1, 0, 0, 1, 0, 1, 0, 1])).all()
    assert (stm.encode(histories[1]) == np.array([0, 1, 1, 0, 0, 1, 1, 0])).all()
    assert (stm.encode(histories[2]) == np.array([1, 0, 1, 0, 1, 0, 0, 1])).all()
    assert (stm.encode(histories[3]) == np.array([0, 1, 0, 1, 1, 0, 0, 1])).all()
