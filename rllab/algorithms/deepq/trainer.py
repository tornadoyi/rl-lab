from rllab import torchlab as tl
from rllab.torchlab import optim, distributed
from rllab import envs
from rllab.rl import features
from rllab.rl.profiling import Profiling, indicator
from rllab.rl.common.schedule import LinearSchedule
from . import replay_buffer
from .deepq import DeepQ


class Trainer(object):
    def __init__(
            self,
            env,
            deepq={},
            rb={},
            explore={},
            optimizer={},
            feature={},
            total_steps=int(1e6),
            learning_starts=1000,
            train_freq=1,
            target_network_update_freq=500,
            profiling={},
            batch_size=32,
            device=None,
            **_,
    ):
        # arguments
        self.total_steps = total_steps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.device = device

        # env
        self.env = envs.make(**env)

        # features
        feature_creator = lambda: features.build(self.env.observation_space, **feature)

        # algorithm
        self.deepq = DeepQ(
            self.env.observation_space,
            self.env.action_space,
            feature_creator,
            **deepq
        ).to(self.device)

        # optimizer
        opt = dict({'name':'Adam', 'lr':1e-3}, **optimizer)
        self.optimizer = optim.build(params=self.deepq.trained_parameters, **opt)

        # replay buffer
        self.replay_buffer = replay_buffer.build(**rb)

        # create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(
            schedule_timesteps=int(explore.get('fraction', 0.1) * total_steps),
            initial_p=explore.get('init', 1.0),
            final_p=explore.get('final', 0.02)
        )

        # steps
        self.steps = 0

        # profiling
        self.profiling = Profiling(self.env, step_func=lambda: self.steps, **profiling)


    def __call__(self, *args, **kwargs):
        ob = self.env.reset()
        while self.steps < self.total_steps:
            self.steps += 1
            # self.env.render()

            # evaluate action
            eps = self.exploration.value(self.steps)
            action = self.deepq.act(
                tl.as_tensor(ob, dtype=tl.float32, device=self.device),
                eps
            ).cpu().data.numpy()

            # exec action
            ob_n, rew, done, _ = self.env.step(action)

            # store transition in the replay buffer.
            self.replay_buffer.add(ob, action, rew, ob_n, float(done))

            # train once
            learn_info = None
            if self.steps > self.learning_starts and self.steps % self.train_freq == 0:
                obs, acs, rews, obs_n, dones = self.replay_buffer.sample(self.batch_size)
                learn_info = self.deepq.learn(
                    self.optimizer,
                    tl.as_tensor(obs, dtype=tl.float32, device=self.device),
                    tl.as_tensor(acs, dtype=tl.long, device=self.device),
                    tl.as_tensor(rews, dtype=tl.float32, device=self.device),
                    tl.as_tensor(obs_n, dtype=tl.float32, device=self.device),
                    tl.as_tensor(dones, dtype=tl.float32, device=self.device),
                    tl.as_tensor([1.0] * obs.shape[0], dtype=tl.float32, device=self.device),
                )

            # update target network
            if self.steps > self.learning_starts and self.steps % self.target_network_update_freq == 0:
                self.deepq.update_target_network()

            # profiling
            self.profile({'explore_epsilon': eps}, learn_info)

            # next
            ob = ob_n
            if done: ob = self.env.reset()


    def profile(self, hps, learn_info):
        p = self.profiling

        # hyper parameters
        for k, v in hps.items():
            p.update('hp/{}'.format(k), v, creator=lambda: indicator('scalar').cond('update', 100))


        # profile learn info
        learn_info = learn_info or {}
        for k, (v, creator) in learn_info.items():
            p.update(k, v, creator=creator)

        # step for profiling
        p()



def train(dist=None, device=None, **kwargs):
    # select device
    device = tl.select_device(device)

    # single process
    if dist is None: return Trainer(device=device, **kwargs)()

    distributed.launch(target=dist_trainer, kwargs={'device':device, 'kwargs': kwargs}, **dist)

# distributed
def dist_trainer(device, kwargs):
    # set device
    dist_device = device
    if device.type == 'gpu':
        index = distributed.get_rank() % tl.cuda.device_count()
        dist_device = tl.device('cuda:{}'.format(index))

    Trainer(device=dist_device, **kwargs)()