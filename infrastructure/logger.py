from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Logger:
    def __init__(self, log_dir, n_logged_samples=10):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):
        videos = [np.transpose(path['image_obs'], [0, 3, 1, 2]) for path in paths]

        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        for i in range(max_videos_to_save):
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(videos, video_title, step, fps=fps)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def flush(self):
        self._summ_writer.flush()
