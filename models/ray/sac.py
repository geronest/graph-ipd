"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

implementation of soft actor critic
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
class RaySACAgent(Agent):
    def __init__(
        self,
        idx_agent,
        env,
        config,
        state_manager,
        name="sac",
        seed_modifier=0,
        verbose=False,
    ):
        super().__init__(idx_agent, env, config, state_manager)
        config_model = config[name]

        if config.device == "gpu":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.name = "sac"
        # fix num_actions to 1 when used for partner selection
        if name == "sac_ps":
            self.num_actions = 1
            self.name += "_ps"

        self.stm = state_manager

        self.sanity_check(config_model)

        # define network
        random_seed = config.random_seeds.agent + seed_modifier
        torch.manual_seed(random_seed)
        self.rng = np.random.default_rng(seed=random_seed)

        # self.layers = dict()
        self.layers = nn.ModuleDict()
        for k in ["value1", "value2", "policy"]:
            self.layers[k] = nn.ModuleList()
            self.layers[k].append(
                nn.Linear(
                    self.dim_state,
                    config_model.dims_hidden_layers[0],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            # hidden layers
            for idx_hl in range(len(config_model.dims_hidden_layers) - 1):
                self.layers[k].append(
                    nn.Linear(
                        config_model.dims_hidden_layers[idx_hl],
                        config_model.dims_hidden_layers[idx_hl + 1],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            # output layer
            self.layers[k].append(
                nn.Linear(
                    config_model.dims_hidden_layers[-1],
                    self.num_actions,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            # self.define_network(config_model, self.layers[k])
        self.layers["target1"] = copy.deepcopy(self.layers["value1"])
        self.layers["target2"] = copy.deepcopy(self.layers["value2"])

        self.learning_rate = config_model.learning_rate

        self.gamma = config_model.gamma
        self.alpha = config_model.alpha
        self.tau = config_model.tau

        self.optim = torch.optim.SGD(
            self.parameters(),
            self.learning_rate,
            weight_decay=config_model.weight_decay,
        )
        self.step_count = 0
        self.freq_refresh_target = config_model.freq_refresh_target
        self.losses = deque(maxlen=config.record_step)
        self.reset_buffer()

        # modify last layer's bias to favour cooperation if conditions satisfy
        if "p_favour_cooperation" in config_model:
            p_favour_cooperation = config_model.p_favour_cooperation
            if self.rng.random() < p_favour_cooperation:
                self.favour_cooperation()

        if verbose:
            print(f"[Soft Actor-Critic: {name}] initialised")

    def sanity_check(self, config):
        assert config.learning_rate >= 0.0
        assert (config.gamma >= 0.0) and (config.gamma <= 1.0)
        assert config.alpha >= 0.0

    def define_network(self, config_model, layers):
        # layers = nn.ModuleList()
        # input layer
        layers.append(
            nn.Linear(
                self.dim_state,
                config_model.dims_hidden_layers[0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # hidden layers
        for idx_hl in range(len(config_model.dims_hidden_layers) - 1):
            layers.append(
                nn.Linear(
                    config_model.dims_hidden_layers[idx_hl],
                    config_model.dims_hidden_layers[idx_hl + 1],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        # output layer
        layers.append(
            nn.Linear(
                config_model.dims_hidden_layers[-1],
                self.num_actions,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # return layers

    def reset_buffer(self):
        """
        reset training buffer.
        """
        self.train_buffer = dict()
        for k in ["states", "actions", "rewards", "nstates"]:
            self.train_buffer[k] = list()

    def store_buffer(self, key, value):
        self.train_buffer[key].append(value)

    def forward_sac(self, layers, x):
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

        if out.isnan().any():
            print(f"[{self.name}-Forward_SAC] isNaN")
            raise ValueError
        return out

    def forward(self, x, key="policy"):
        if key == "policy":
            if "ps" in self.name:
                return torch.softmax(
                    self.forward_sac(self.layers[key], x), axis=-2
                ).reshape(x.shape[0], -1)
            else:
                return torch.softmax(
                    self.forward_sac(self.layers[key], x), dim=-1
                ).reshape(x.shape[0], -1)
        else:
            return self.forward_sac(self.layers[key], x).reshape(x.shape[0], -1)

    def add_name(self, name):
        """
        add a string to the agent's name.
        * made for partner selection module, pairing it with different types of agents
        """
        self.name += "_" + name

    def sample_policy(self, state, cand_agents=None):
        with torch.no_grad():
            # sample action
            if "ps" in self.name:
                probs = self(state[cand_agents]).reshape(-1).numpy()
                action = self.rng.choice(cand_agents, 1, p=probs)[0]
            else:
                probs = self(state)
                action = torch.multinomial(probs, 1)[0].item()
        return action

    def act(self, state, is_me=False):
        """
        given a state information, return action of the agent.
        Arguments:
            state
        Returns:
            action
        """
        action = self.sample_policy(state)

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

            # obtain policy distribution
            if "ps" in self.name:
                ps_states = np.array(self.train_buffer["nstates"])
                probs_policy = self(ps_states)
                qvals = torch.min(
                    self(ps_states, key="value1"), self(ps_states, key="value2")
                )
            else:
                probs_policy = self(states)
                qvals = torch.min(
                    self(states, key="value1"), self(states, key="value2")
                )

            # update value network
            self.optim.zero_grad(set_to_none=True)

            oldq1 = self(states, "value1").gather(1, actions)
            oldq2 = self(states, "value2").gather(1, actions)
            with torch.no_grad():
                next_probs_policy = self(next_states)
                next_minq = torch.min(
                    self(next_states, key="target1"), self(next_states, key="target2")
                )
                y = rewards + self.gamma * (
                    next_probs_policy
                    * (next_minq - self.alpha * torch.log(next_probs_policy))
                ).sum(axis=-1)

            # loss_q = torch.mean((oldq1 - y.detach()) ** 2 + (oldq2 - y.detach()) ** 2)
            loss_q1 = torch.mean((oldq1 - y.detach()) ** 2)
            loss_q2 = torch.mean((oldq2 - y.detach()) ** 2)
            self.losses.append(loss_q1.item())

            # update policy network
            loss_pi = torch.mean(
                (
                    probs_policy
                    * (self.alpha * torch.log(probs_policy) - qvals.detach()) 
                ).sum(axis=-1)
            )

            # loss_q.backward()
            loss_q1.backward()
            loss_q2.backward()
            loss_pi.backward()
            self.optim.step()

            # update target network
            self.step_count += 1
            for idx_net in range(1, 3):
                k_target = f"target{idx_net}"
                k_value = f"value{idx_net}"
                for idx_layer in range(len(self.layers[k_target])):
                    for t, s in zip(
                        self.layers[k_target][idx_layer].parameters(),
                        self.layers[k_value][idx_layer].parameters(),
                    ):
                        t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

            if (
                self.config.print_loss
                and (self.step_count % self.config.record_step) == 0
            ):
                avg_loss = np.mean(self.losses)
                print(
                    "[SACAgent-step_train()] average loss across"
                    + f"recent {self.config.record_step} steps: "
                    + f"{avg_loss:.4f}"
                )
                print(f"[SACAgent-step_train()] Q-values: {self(states)}")

            self.reset_buffer()

    def select_partner(self, cand_agents, states_op):
        """
        select partner given their states.
        """
        if "ps" in self.name:
            action = self.sample_policy(states_op, cand_agents)

        self.store_buffer("states", states_op[action])
        self.store_buffer("actions", 0)
        self.store_buffer("nstates", states_op[cand_agents])
        # self.store("states", states_op[action])

        return action

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
