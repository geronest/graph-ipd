import numpy as np
from omegaconf import OmegaConf

from env.agentnet import AgentNetwork, check_connectedness
from env.matgame import MatrixGameEnv
from utils.configs import init_config  # noqa: E402


def test_matrix_game():
    config = OmegaConf.load("./configs/test/default.yaml")
    config = init_config(config)
    game = MatrixGameEnv(config.matgame)
    new_states = game.reset()
    assert (new_states == np.array([0, 0])).all()
    assert (game.step((0, 0))[0] == np.array([1, 1])).all()
    assert (game.step((1, 0))[0] == np.array([0, 3])).all()
    assert (game.step((0, 1))[0] == np.array([3, 0])).all()
    assert (game.step((1, 1))[0] == np.array([2, 2])).all()

    actions = (0, 1)
    rewards, new_states = game.step(actions)
    assert rewards[0] == 3
    assert rewards[1] == 0
    assert (new_states == np.array([0, 0])).all()


def test_agent_network():
    config = OmegaConf.load("./configs/test/default.yaml")
    config = init_config(config)
    anet = AgentNetwork(config, 10)

    for i in range(100):
        idxs = anet.sample_agents()
        assert anet.network[idxs[0], idxs[1]] > 0


def test_check_connectedness():
    m1 = np.eye(3)
    m2 = np.ones((3, 3))
    m3 = m2 - m1
    m4 = np.array([[0, 1], [1, 0]])
    m5 = np.array([[1, 1], [0, 0]])
    m6 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    m7 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    assert not check_connectedness(m1)
    assert check_connectedness(m2)
    assert check_connectedness(m3)
    assert check_connectedness(m4)
    assert check_connectedness(m5)
    assert not check_connectedness(m6)
    assert check_connectedness(m7)
