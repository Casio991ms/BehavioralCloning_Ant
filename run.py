import os
import time

from agents.bc_agent import BCAgent
from agents.loaded_gaussian_policy import LoadedGaussianPolicy
from infrastructure.trainer import Trainer
from infrastructure.utils import MJ_ENV_NAMES, MJ_ENV_KWARGS


class BC_Trainer(object):
    def __init__(self, params):
        agent_params = {
            'n_hidden_layers': params['n_hidden_layers'],
            'hidden_size': params['hidden_size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
        }

        self.params = params
        self.params['agent_class'] = BCAgent
        self.params['agent_params'] = agent_params

        self.params["env_kwargs"] = MJ_ENV_KWARGS[self.params['env_name']]

        self.trainer = Trainer(self.params)

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):
        self.trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            collect_policy=self.trainer.agent.policy,
            eval_policy=self.trainer.agent.policy,
            initial_expertdata=self.params['expert_data'],
            expert_policy=self.loaded_expert_policy
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str,
                        required=True)
    parser.add_argument('--expert_data', '-ed', type=str,
                        required=True)
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int,
                        default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int,
                        default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_hidden_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--hidden_size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    params = vars(args)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    trainer = BC_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
