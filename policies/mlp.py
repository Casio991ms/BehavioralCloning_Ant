import itertools

import torch
from torch import nn, optim
from infrastructure import pytorch_utils as ptu

import numpy as np


class MLPPolicy(nn.Module):
    def __init__(self,
                 ob_dim,
                 n_hidden_layers,
                 hidden_size,
                 ac_dim,
                 learning_rate=1e-4,
                 ):

        super().__init__()
        self.ob_dim = ob_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.ac_dim = ac_dim
        self.learning_rate = learning_rate

        self.logits_na = None
        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim,
            n_hidden_layers=self.n_hidden_layers,
            hidden_size=self.hidden_size,
            output_size=self.ac_dim,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )
        self.loss = nn.MSELoss()

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        return ptu.to_numpy(self(ptu.from_numpy(observation)))

    def update(self, observations, actions, **kwargs):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        self.optimizer.zero_grad()
        logits = self(observations)
        loss = self.loss(logits, actions)
        loss.backward()
        self.optimizer.step()
        return {'Training Loss': ptu.to_numpy(loss)}

    def forward(self, obs):
        return self.mean_net(obs)
