"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of deep q network
interfaces based on openAI gym
"""

import copy
from collections import deque

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..agent import Agent


@ray.remote(num_gpus=0.0005, num_cpus=0.01)
class RayDQNAgent(Agent):
    def __init__(
        self,
        idx_agent,
        env,
        config,
        state_manager,
        name="dqn",
        seed_modifier=0,
        verbose=False,
    ):
        super().__init__(idx_agent, env, config, state_manager)
        config_model = config[name]

        if config.device == "gpu":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.name = "dqn"
        # fix num_actions to 1 when used for partner selection
        if name == "dqn_ps":
            self.num_actions = 1
            self.name += "_ps"

        self.stm = state_manager

        self.sanity_check(config_model)

        # define network
        random_seed = config.random_seeds.agent + seed_modifier
        torch.manual_seed(random_seed)
        self.rng = np.random.default_rng(seed=random_seed)

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            nn.Linear(
                self.dim_state,
                config_model.dims_hidden_layers[0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # hidden layers
        for idx_hl in range(len(config_model.dims_hidden_layers) - 1):
            self.layers.append(
                nn.Linear(
                    config_model.dims_hidden_layers[idx_hl],
                    config_model.dims_hidden_layers[idx_hl + 1],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        # output layer
        self.layers.append(
            nn.Linear(
                config_model.dims_hidden_layers[-1],
                self.num_actions,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.layers_target = copy.deepcopy(self.layers)

        self.learning_rate = config_model.learning_rate

        # exploration scheme
        if "exploration" in config_model:
            self.exploration_scheme = config_model["exploration"]
            self.softmax = nn.Softmax(dim=0)
            if "exploration_temperature" in config_model:
                self.exploration_temperature = config_model["exploration_temperature"]
            else:
                self.exploration_temperature = None
        else:
            self.exploration_scheme = "epsilon_greedy"
        self.epsilon = nn.Parameter(
            torch.tensor(config_model.epsilon, dtype=torch.float, device=self.device),
            requires_grad=False,
        )
        self.prepare_epsilon_decay(config, name)
        self.gamma = config_model.gamma

        self.optim = torch.optim.SGD(
            self.parameters(),
            self.learning_rate,
            weight_decay=config_model.weight_decay,
        )
        self.step_count = 0
        self.freq_refresh_target = config_model.freq_refresh_target
        self.losses = deque(maxlen=config.record_step)
        self.cached_Q = None
        self.reset_buffer()

        # modify last layer's bias to favour cooperation if conditions satisfy
        if "p_favour_cooperation" in config_model:
            p_favour_cooperation = config_model.p_favour_cooperation
            if self.rng.random() < p_favour_cooperation:
                self.favour_cooperation()

        if verbose:
            print(f"[Deep Q-learning: {name}] initialised")

    def sanity_check(self, config):
        assert config.learning_rate >= 0.0
        assert (config.epsilon >= 0.0) and (config.epsilon <= 1.0)
        assert (config.gamma >= 0.0) and (config.gamma <= 1.0)
        assert isinstance(config.freq_refresh_target, int) and (
            config.freq_refresh_target > 0
        )

    def prepare_epsilon_decay(self, config, name):
        """
        set parameters for epsilon decay.
        """
        self.epsilon_decay = dict()
        if "epsilon_decay" in config[name]:
            config_model = config[name]
            self.epsilon_decay["end_step"] = (
                config.num_steps * config_model["epsilon_decay"]["end_step"]
            )
            target_eps = config_model["epsilon_decay"]["target_eps"]
            initial_eps = config_model["epsilon"]
            self.epsilon_decay["decay_rate"] = np.exp(
                np.log(target_eps / initial_eps) / self.epsilon_decay["end_step"]
            )

    def reset_buffer(self):
        """
        reset training buffer.
        """
        self.train_buffer = dict()
        for k in ["states", "actions", "rewards", "nstates"]:
            self.train_buffer[k] = list()

    def store_buffer(self, key, value):
        self.train_buffer[key].append(value)

    def favour_cooperation(self):
        """
        modify the last layer's bias value to make it prefer cooperation.
        """
        sdict = self.state_dict()
        ks = list(sdict.keys())
        idx_lastlayer = -1
        for k in ks:
            if k[:6] == "layers":
                idx_layer = int(k.split(".")[1])
                if idx_layer > idx_lastlayer:
                    idx_lastlayer = idx_layer

        w = sdict[f"layers.{idx_lastlayer}.weight"].numpy()
        b = sdict[f"layers.{idx_lastlayer}.bias"].numpy()
        nw = abs(w).sum(-1)
        nw[0] *= -1
        nb = abs(b)
        nb[0] *= -1
        new_bias = torch.tensor(nw + nb)
        sdict[f"layers.{idx_lastlayer}.bias"] = new_bias
        sdict[f"layers_target.{idx_lastlayer}.bias"] = new_bias

        self.load_state_dict(sdict)

    def forward_dqn(self, layers, x):
        """
        forward one of {training, target} networks.
        Arguments:
            layers
            x
        Returns:
            network output given x
        """
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        for layer in layers[:-1]:
            x = F.relu(layer(x))
        out = layers[-1](x)

        # if out.isnan().any() or self.rng.random() < 1e-3:
        if out.isnan().any():
            print(f"[{self.name}-Forward_DQN] isNaN")
            raise ValueError
        return out

    def forward(self, x):
        return self.forward_dqn(self.layers, x)

    def add_name(self, name):
        """
        add a string to the agent's name.
        * made for partner selection module, pairing it with different types of agents
        """
        self.name += "_" + name

    def act(self, state, is_me=False):
        """
        given a state information, return action of the agent.
        Arguments:
            state
        Returns:
            action
        """
        with torch.no_grad():
            # apply epsilon
            if (
                torch.rand((1), device=torch.device(self.device)) < self.epsilon
            ):  # random action
                action = int(torch.rand((1)) * self.num_actions)
            else:  # choose action with highest q value
                action = self(state).argmax().item()

        self.store_buffer("states", state)
        self.store_buffer("actions", action)
        self.store_buffer("nstates", state)

        if is_me or self.selected_history:
            self.store("states", state)

        return action

    def step_train(self, new_state):
        """
        single step of the training process.
        """

        if len(self.train_buffer["states"]) > 0:
            states = np.array(self.train_buffer["states"])
            next_states = np.array(self.train_buffer["nstates"][1:] + [new_state])

            actions = torch.tensor(
                self.train_buffer["actions"], dtype=torch.int64, device=self.device
            ).reshape(-1, 1)
            rewards = torch.tensor(
                self.train_buffer["rewards"], dtype=torch.float32, device=self.device
            )

            self.optim.zero_grad(set_to_none=True)

            oldq = self(states).gather(1, actions)
            with torch.no_grad():
                nextq = (
                    self.forward_dqn(self.layers_target, next_states)
                    .reshape(next_states.shape[0], -1)
                    .max(axis=-1)
                    .values
                )
                newq = rewards + self.gamma * nextq

            loss = torch.mean((newq - oldq) ** 2)  # TODO: Exponential Moving Average?
            self.losses.append(loss.item())

            loss.backward()
            self.optim.step()

            # update target network
            self.step_count += 1
            if self.step_count % self.freq_refresh_target == 0:
                for idx_layer in range(len(self.layers_target)):
                    self.layers_target[idx_layer].load_state_dict(
                        self.layers[idx_layer].state_dict()
                    )

            # epsilon decay
            if ("decay_rate" in self.epsilon_decay) and (
                self.step_count < self.epsilon_decay["end_step"]
            ):
                self.epsilon.copy_(self.epsilon * self.epsilon_decay["decay_rate"])

            if (
                self.config.print_loss
                and (self.step_count % self.config.record_step) == 0
            ):
                avg_loss = np.mean(self.losses)
                print(
                    "[DQNAgent-step_train()] average loss across"
                    + f"recent {self.config.record_step} steps: "
                    + f"{avg_loss:.4f}"
                )
                print(f"[DQNAgent-step_train()] Q-values: {self(states)}")

            self.reset_buffer()

    def select_partner(self, cand_agents, states_op):
        """
        select partner given their states.
        """
        if "ps" in self.name:
            with torch.no_grad():
                if self.exploration_scheme == "epsilon_greedy":
                    # apply epsilon
                    if (
                        torch.rand((1), device=torch.device(self.device)) < self.epsilon
                    ):  # random action
                        idx_sample = int(torch.rand((1)) * len(cand_agents))
                        action = cand_agents[idx_sample]
                    else:  # choose action with highest q value
                        qvalues = self(states_op[cand_agents]).numpy().reshape(-1)
                        agents_maxq = cand_agents[qvalues == qvalues.max()]
                        idx_sample = int(torch.rand((1)) * len(agents_maxq))
                        action = agents_maxq[idx_sample]
                elif self.exploration_scheme == "softmax":
                    qvalues = self(states_op[cand_agents]).reshape(-1)
                    if self.exploration_temperature is not None:
                        qvalues = qvalues / self.exploration_temperature
                    p_softmax = self.softmax(qvalues).numpy()
                    action = self.rng.choice(cand_agents, 1, p=p_softmax)[0]
                else:
                    print("ERROR?")
                    raise ValueError

        self.store_buffer("states", states_op[action])
        self.store_buffer("actions", 0)
        self.store_buffer("nstates", states_op[cand_agents])
        # self.store("states", states_op[action])

        return action

    def select_partner2(self, cand_agents, states_op):
        """
        select partner given their states.
        * Uses dilemma-playing NN, no separate NN needed.
        * no epsilon needed, as it requires no learning NN of its own.
        """
        with torch.no_grad():
            # choose action with highest q value
            qvalues = (
                self(states_op[cand_agents]).max(axis=-1).values.numpy().reshape(-1)
            )
            agents_maxq = cand_agents[qvalues == qvalues.max()]
            idx_sample = int(torch.rand((1)) * len(agents_maxq))
            action = agents_maxq[idx_sample]

        return action

    def rewire(self, neighbours, states_op):
        """
        perform rewiring.
        use dilemma-playing network.
        """
        res = {"idx_me": self.idx_me, "attach": None, "detach": None}

        allbutme = np.delete(np.arange(states_op.shape[0]), self.idx_me)
        nonneighbours = np.setdiff1d(allbutme, neighbours)

        with torch.no_grad():
            qvalues = self(states_op[allbutme]).max(axis=-1).values.numpy().reshape(-1)
        change_neighbours = 0

        # 1: attachment
        nodes_maxq = allbutme[qvalues == qvalues.max()]
        # filter neighbours from attachment candidates
        nodes_maxq = np.setdiff1d(nodes_maxq, neighbours)
        if nodes_maxq.shape[0] > 0:
            res["attach"] = nodes_maxq[int(torch.rand((1)) * len(nodes_maxq))]
            change_neighbours += 1

        # 2: detachment
        nodes_minq = allbutme[qvalues == qvalues.min()]
        # filter non-neighbours from detachment candidates
        nodes_minq = np.setdiff1d(nodes_minq, nonneighbours)
        can_detach = (len(neighbours) > 1) or (change_neighbours > 0)
        if (nodes_minq.shape[0] > 0) and can_detach:
            res["detach"] = nodes_minq[int(torch.rand((1)) * len(nodes_minq))]
            change_neighbours -= 1

        return res

    def train(self):
        """
        run training.
        """
        pass

    def save(self, path):
        """
        save model parameters at the designated path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        load model parameters from the designated path.
        """
        self.load_state_dict(torch.load(path))
