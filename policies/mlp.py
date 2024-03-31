import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 n_hidden_layers,
                 hidden_size,
                 learning_rate=1e-4,
                 ):

        super(MLPPolicy, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.logits_na = None
        # self.mean_net = ptu

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    # def get_action(self, obs):
    #     if len(obs.shape) > 1: