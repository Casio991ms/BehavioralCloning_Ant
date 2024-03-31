import numpy as np

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
MJ_ENV_KWARGS["Ant-v4"]["use_contact_forces"] = True


def sample_trajectory(env, policy, max_path_length, render=False):
    ob = env.reset()

    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        if render:
            if hasattr(env, 'sim'):
                image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
            else:
                image_obs.append(env.render())

        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        ob, rew, terminated, truncated, _ = env.step(ac)

        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        rollout_done = 1 if (terminated or truncated or steps >= max_path_length) else 0
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        paths.append(sample_trajectory(env, policy, max_path_length, render))
        timesteps_this_batch += get_pathlength(paths[-1])

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    paths = []

    for _ in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length, render))

    return paths


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    if image_obs:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation": np.array(obs, dtype=np.float32),
            "image_obs": np.array(image_obs, dtype=np.uint8),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def get_pathlength(path):
    return len(path["reward"])
