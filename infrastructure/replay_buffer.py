import numpy as np

from infrastructure.utils import convert_listofrollouts


class ReplayBuffer(object):
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.paths = []

        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        return self.obs.shape[0] if self.obs else 0

    def add_rollouts(self, paths, concat_rew=True):
        for path in paths:
            self.paths.append(path)

        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

    def sample_random_data(self, batch_size):
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.rews.shape[0]
                == self.next_obs.shape[0]
                == self.terminals.shape[0]
        )

        idxs = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[idxs], self.acs[idxs], self.rews[idxs], self.next_obs[idxs], self.terminals[idxs]
