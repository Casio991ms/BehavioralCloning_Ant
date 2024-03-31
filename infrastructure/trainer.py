import time
from collections import OrderedDict

import numpy as np
import torch
import gymnasium as gym

from infrastructure import utils
from infrastructure.logger import Logger

MAX_NVIDEO = 1
MAX_VIDEO_LEN = 100


class Trainer(object):
    def __init__(self, params):
        self.initial_return = None
        self.log_metrics = None
        self.start_time = None
        self.total_envsteps = None
        self.log_video = None
        self.params = params
        self.logger = Logger(self.params['logdir'])

        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.params['video_log_freq'] == -1:
            self.params['env_kwargs']['render_mode'] = None
        self.env = gym.make(self.params['env_name'], **self.params['env_kwargs'])
        self.env.reset(seed=seed)

        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        self.params['agent_params']['ob_dim'] = self.env.observation_space.shape[0]
        self.params['agent_params']['ac_dim'] = self.env.action_space.shape[0]

        self.fps = self.env.env.metadata['render_fps']

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None):

        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)

            self.log_video = (self.params['video_log_freq'] != -1 and itr % self.params['video_log_freq'] == 0)

            self.log_metrics = (itr % self.params['scalar_log_freq'] == 0)

            paths, envsteps_this_batch, train_video_paths = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy,
                self.params['batch_size']
            )
            self.total_envsteps += envsteps_this_batch

            self.agent.add_to_replay_buffer(paths)

            training_logs = self.train_agent()

            if self.log_video or self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)

                if self.params['save_params']:
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata,
            collect_policy,
            batch_size,
    ):

        if itr == 0:
            loaded_paths = np.load(load_initial_expertdata, allow_pickle=True)
            return loaded_paths, 0, None

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env, collect_policy, batch_size, self.params['ep_len'], False
        )

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            print(f"len(train_video_paths): {len(train_video_paths)}")
            print(f"train_video_paths[0].keys(): {train_video_paths[0].keys()}")
            print(f"train_video_paths[0]['image_obs'].shape: {train_video_paths[0]['image_obs'].shape}")

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])

            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs


    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy,
                                                                         self.params['eval_batch_size'],
                                                                         self.params['ep_len'])

        if self.log_video and train_video_paths is not None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        if self.log_metrics:
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()


