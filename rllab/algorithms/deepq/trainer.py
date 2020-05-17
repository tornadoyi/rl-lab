from rllab.torchlab import optim
from rllab import envs
from rllab.rl.profiling import Profiling
from rllab.rl.common.schedule import LinearSchedule
from . import replay_buffer
from .deepq import DeepQ


class Trainer(object):
    def __init__(
            self,
            env,
            alg={},
            rb={},
            explore={},
            optimizer={},
            total_steps=100000,
            learning_starts=1000,
            train_freq=1,
            target_network_update_freq=500,
            profiling={},
            batch_size=32,
            **_,
    ):
        # arguments
        self.total_steps = total_steps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq

        # env
        self.env = envs.make(**env)

        # algorithm
        self.deepq = DeepQ(
            self.env.observation_space,
            self.env.action_space,
            **alg
        )

        # optimizer
        opt = dict({'name':'Adam', 'lr':1e-3}, **optimizer)
        self.optimizer = optim.build(params=self.deepq.parameters, **opt)

        # replay buffer
        self.replay_buffer = replay_buffer.build(**rb)

        # create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(
            schedule_timesteps=int(explore.get('fraction', 0.1) * total_steps),
            initial_p=explore.get('init', 0.1),
            final_p=explore.get('final', 0.02)
        )

        # steps
        self.steps = 0

        # profiling
        self.profiling = Profiling(self.env, step_func=lambda: self.steps, **dict({'flush_freq': 1, 'print_freq': 1}, **profiling))


    def __call__(self, *args, **kwargs):
        ob = self.env.reset()
        while self.steps < self.total_steps:
            self.steps += 1
            # self.env.render()

            # evaluate action
            eps = self.exploration.value(self.steps)
            action = self.deepq.act(ob, eps)

            # exec action
            ob_n, rew, done, _ = self.env.step(action)

            # store transition in the replay buffer.
            self.replay_buffer.add(ob, action, rew, ob_n, float(done))

            # train once
            learn_info = None
            if self.steps > self.learning_starts and self.steps % self.train_freq == 0:
                obs, acs, rews, obs_n, dones = self.replay_buffer.sample(self.batch_size)
                learn_info = self.deepq.learn(self.optimizer, obs, acs, rews, obs_n, dones)

            # update target network
            if self.steps > self.learning_starts and self.steps % self.target_network_update_freq == 0:
                self.deepq.update_target_network()

            # profiling
            self.profile(learn_info)

            # next
            ob = ob_n
            if done: ob = self.env.reset()


    def profile(self, learn_info):
        p = self.profiling

        # profile learn info
        learn_info = learn_info or {}
        for k, (v, creator) in learn_info.items():
            p.update(k, v, creator=creator)

        # step for profiling
        p()