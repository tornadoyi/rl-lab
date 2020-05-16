from rllab.torchlab import optim
from rllab import envs
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
            optimizer={'name': 'Adam', 'lr': 1e-3},
            total_steps=100000,
            learning_starts=1000,
            train_freq=1,
            target_network_update_freq=500,
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
        self.optimizer = optim.build(params=self.deepq.parameters, **optimizer)

        # replay buffer
        self.replay_buffer = replay_buffer.build(**rb)

        # create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(
            schedule_timesteps=int(explore.get('fraction', 0.1) * total_steps),
            initial_p=explore.get('init', 0.1),
            final_p=explore.get('final', 0.02)
        )




    def __call__(self, *args, **kwargs):
        ob = self.env.reset()
        for t in range(self.total_steps):
            # evaluate action
            eps = self.exploration.value(t)
            action = self.deepq.act(ob, eps)

            # exec action
            ob_n, rew, done, _ = self.env.step(action)

            # store transition in the replay buffer.
            self.replay_buffer.add(ob, action, rew, ob_n, float(done))

            # next
            ob = ob_n
            if done: ob = self.env.reset()

            #  train once
            if t > self.learning_starts and t % self.train_freq == 0:
                obs, acs, rews, obs_n, dones = self.replay_buffer.sample(self.batch_size)
                self.deepq.learn(self.optimizer, obs, acs, rews, obs_n, dones)

            # update target network
            if t > self.learning_starts and t % self.target_network_update_freq == 0:
                self.deepq.update_target_network()