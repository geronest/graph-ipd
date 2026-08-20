import os
import shutil

import ray
from omegaconf import OmegaConf

from env.matgame import MatrixGameEnv
from env.statemanager import StateManager
from models.dqn import DQNAgent
from models.ray.dqn import RayDQNAgent
from models.tabularq import TabularQLearningAgent
from utils.configs import init_config


def test_tabularq():
    config = OmegaConf.load("./configs/test/default.yaml")
    config = init_config(config)
    game = MatrixGameEnv(config.matgame)
    stm = StateManager(config)
    states = game.reset()

    test_dir = "./results/test_models/test_tabularq/run0/models/iter0"
    os.makedirs(test_dir, exist_ok=True)

    agent1 = TabularQLearningAgent(0, game, config, stm, seed_modifier=1)
    agent2 = TabularQLearningAgent(1, game, config, stm, seed_modifier=2)
    agent1.store("actions", [1, 0])
    agent2.store("actions", [0, 1])
    states = [
        stm.encode(agent1.history),
        stm.encode(agent2.history),
    ]

    # learning
    for i in range(100):
        action1 = agent1.act(states[0])
        action2 = agent2.act(states[1])
        agent1.store("actions", [action1, action2])
        agent2.store("actions", [action2, action1])

        rewards, new_states = game.step((action1, action2))
        new_states = [
            stm.encode(agent1.history),
            stm.encode(agent2.history),
        ]

        agent1.step_train(states[0], action1, rewards[0], new_states[0])
        agent2.step_train(states[1], action2, rewards[1], new_states[1])

        states = new_states

    agent1.save(test_dir + "agent1.pth")

    agent3 = TabularQLearningAgent(0, game, config, stm, seed_modifier=3)
    agent3.load(test_dir + "agent1.pth")

    assert str(agent1.state_dict()) == str(agent3.state_dict())

    shutil.rmtree("./results/test_models/")


def test_dqn():
    config = OmegaConf.load("./configs/test/default.yaml")
    config = init_config(config)
    game = MatrixGameEnv(config.matgame)
    stm = StateManager(config)
    states = game.reset()

    test_dir = "./results/test_models/test_dqn/run0/models/iter0"
    os.makedirs(test_dir, exist_ok=True)

    agent1 = DQNAgent(0, game, config, stm, seed_modifier=1)
    agent2 = DQNAgent(1, game, config, stm, seed_modifier=2)
    agent1.store("actions", [1, 0])
    agent2.store("actions", [0, 1])
    states = [
        stm.encode(agent1.history),
        stm.encode(agent2.history),
    ]

    # learning
    for i in range(100):
        action1 = agent1.act(states[1], prepare_train=True)
        action2 = agent2.act(states[0], prepare_train=True)
        agent1.store("actions", [action1, action2])
        agent2.store("actions", [action2, action1])

        rewards, new_states = game.step((action1, action2))
        new_states = [
            stm.encode(agent1.history),
            stm.encode(agent2.history),
        ]

        agent1.step_train(states[1], action1, rewards[0], new_states[1])
        agent2.step_train(states[0], action2, rewards[1], new_states[0])

        states = new_states

    agent1.save(test_dir + "agent1.pth")

    agent3 = DQNAgent(0, game, config, stm, seed_modifier=3)
    agent3.load(test_dir + "agent1.pth")

    assert str(agent1.state_dict()) == str(agent3.state_dict())

    shutil.rmtree("./results/test_models/")


def test_raydqn():
    config = OmegaConf.load("./configs/test/default.yaml")
    config = init_config(config)
    game = MatrixGameEnv(config.matgame)
    stm = StateManager(config)
    states = game.reset()

    test_dir = "./results/test_models/test_dqn/run0/models/iter0"
    os.makedirs(test_dir, exist_ok=True)

    agent1 = RayDQNAgent.remote(0, game, config, stm, seed_modifier=1)
    agent2 = RayDQNAgent.remote(1, game, config, stm, seed_modifier=2)
    agent1.store.remote("actions", [1, 0])
    agent2.store.remote("actions", [0, 1])
    states = [
        stm.encode(ray.get(agent1.getattr.remote("history"))),
        stm.encode(ray.get(agent2.getattr.remote("history"))),
    ]

    # learning
    # Mirrors the live loop in bin/train_ray.py: act(state, is_me) buffers the
    # state and action, rewards are pushed with store_buffer, and step_train()
    # takes only the next state.
    for i in range(100):
        action1 = ray.get(agent1.act.remote(states[1], True))
        action2 = ray.get(agent2.act.remote(states[0], True))
        ray.get(agent1.store.remote("actions", [action1, action2]))
        ray.get(agent2.store.remote("actions", [action2, action1]))

        rewards, new_states = game.step((action1, action2))
        new_states = [
            stm.encode(ray.get(agent1.getattr.remote("history"))),
            stm.encode(ray.get(agent2.getattr.remote("history"))),
        ]

        ray.get(agent1.store_buffer.remote("rewards", rewards[0]))
        ray.get(agent2.store_buffer.remote("rewards", rewards[1]))

        ray.get(agent1.step_train.remote(new_states[1]))
        ray.get(agent2.step_train.remote(new_states[0]))

        states = new_states

    agent1.save.remote(test_dir + "agent1.pth")

    agent3 = RayDQNAgent.remote(0, game, config, stm, seed_modifier=3)
    ray.get(agent3.load.remote(test_dir + "agent1.pth"))

    assert str(ray.get(agent1.state_dict.remote())) == str(
        ray.get(agent3.state_dict.remote())
    )

    shutil.rmtree("./results/test_models/")
