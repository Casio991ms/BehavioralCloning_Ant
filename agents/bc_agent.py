from infrastructure.replay_buffer import ReplayBuffer
from policies.mlp import MLPPolicy


class BCAgent:
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.policy = MLPPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = self.policy.update(ob_no, ac_na)
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)

    def save(self, path):
        return self.policy.save(path)
