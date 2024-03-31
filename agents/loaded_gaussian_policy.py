import pickle

import torch.nn as nn
import numpy as np

from infrastructure import pytorch_utils as ptu


def create_linear_layer(W, b) -> nn.Linear:
    out_features, in_features = W.shape
    linear_layer = nn.Linear(
        in_features,
        out_features,
    )
    linear_layer.weight.data = ptu.from_numpy(W.T)
    linear_layer.bias.data = ptu.from_numpy(b[0])
    return linear_layer


def read_layer(l):
    assert list(l.keys()) == ['AffineLayer']
    assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
    return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer'][
        'b'].astype(np.float32)


class LoadedGaussianPolicy(nn.Module):
    def __init__(self, filename, **kwargs):
        super().__init__()

        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        self.nonlin_type = data['nonlin_type']
        if self.nonlin_type == 'lrelu':
            self.non_lin = nn.LeakyReLU(0.01)
        elif self.nonlin_type == 'tanh':
            self.non_lin = nn.Tanh()
        else:
            raise NotImplementedError()
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', (
            'Policy type {} not supported'.format(policy_type)
        )
        self.policy_params = data[policy_type]

        assert set(self.policy_params.keys()) == {
            'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'
        }

        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = self.policy_params['obsnorm']['Standardizer'][
            'meansq_1_D']
        obsnorm_stdev = np.sqrt(
            np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)

        self.obs_norm_mean = nn.Parameter(ptu.from_numpy(obsnorm_mean))
        self.obs_norm_std = nn.Parameter(ptu.from_numpy(obsnorm_stdev))
        self.hidden_layers = nn.ModuleList()

        # Hidden layers next
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            W, b = read_layer(layer_params[layer_name])
            linear_layer = create_linear_layer(W, b)
            self.hidden_layers.append(linear_layer)

        # Output layer
        W, b = read_layer(self.policy_params['out'])
        self.output_layer = create_linear_layer(W, b)
