import numpy as np


from gym.envs.registration import register


register(
    id='ShuffleRun-100m-v0',
    entry_point='rllab.envs.race:ShuttleRunEnv',
    kwargs={'length': 100, 'move_success_rate': 0.8},
    max_episode_steps=np.inf,
    reward_threshold=1.0,
)