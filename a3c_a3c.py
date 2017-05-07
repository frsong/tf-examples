"""
Asynchronous advantage actor-critic based on

    https://github.com/openai/universe-starter-agent

Original paper:

    Asynchronous methods for deep reinforcement learning.
    https://arxiv.org/abs/1602.01783

"""
import queue
from collections import namedtuple
from threading import Thread
import numpy as np
import tensorflow as tf
import scipy.signal

from a3c_model import Policy

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

Batch = namedtuple('Batch', ('obs', 'a', 'adv', 'r', 'done', 'state'))

def process_rollout(rollout, gamma=0.99, lambda_=1.0):
    """
    Compute the returns and advantages for this rollout.

    """
    batch_obs = np.asarray(rollout.observations)
    batch_a   = np.asarray(rollout.actions)
    rewards   = np.asarray(rollout.rewards)
    vpred_t   = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r        = discount(rewards_plus_v, gamma)[:-1]
    delta_t        = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv      = discount(delta_t, gamma * lambda_)

    state = rollout.states[0]

    return Batch(batch_obs, batch_a, batch_adv, batch_r, rollout.done, state)

class Rollout(object):
    def __init__(self):
        self.observations = []
        self.actions      = []
        self.rewards      = []
        self.values       = []
        self.states       = []

        self.r    = 0.0
        self.done = False

    def add(self, obs, action, reward, value, done, state):
        self.observations += [obs]
        self.actions      += [action]
        self.rewards      += [reward]
        self.values       += [value]
        self.states       += [state]

        self.done = done

    def extend(self, rollout):
        assert not self.done

        self.observations += rollout.observations
        self.actions      += rollout.actions
        self.rewards      += rollout.rewards
        self.values       += rollout.values
        self.states       += rollout.states

        self.r    = rollout.r
        self.done = rollout.done

class RunnerThread(Thread):
    def __init__(self, env, policy, num_local_steps=20):
        super(RunnerThread, self).__init__()
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_state = None
        self.policy = policy
        self.daemon = True
        self.sess   = None
        self.summary_writer = None

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            rollout_provider = env_runner(self.env,
                                          self.policy,
                                          self.num_local_steps,
                                          self.summary_writer)
            while True:
                # Original code says the timeout is needed to make the
                # workers die together
                self.queue.put(next(rollout_provider), timeout=600.0)

def env_runner(env, policy, num_local_steps, summary_writer):
    last_obs   = env.reset()
    last_state = policy.get_initial_state()

    length  = 0
    rewards = 0
    while True:
        terminal_end = False
        rollout = Rollout()
        for _ in range(num_local_steps):
            # Run policy for one step
            action, value, state = policy.act(last_obs, last_state)

            # Perform the action
            obs, reward, done, info = env.step(action.argmax())

            # Add to rollout
            rollout.add(last_obs, action, reward, value, done, last_state)
            length  += 1
            rewards += reward

            # Update observation and state
            last_obs   = obs
            last_state = state

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            tag = 'wrapper_config.TimeLimit.max_episode_steps'
            max_length = env.spec.tags.get(tag)
            if done or length >= max_length:
                terminal_end = True
                if (length >= max_length
                    or not env.metadata.get('semantics.autoreset')):
                    last_obs = env.reset()
                last_state = policy.get_initial_state()
                length  = 0
                rewards = 0
                break
        if not terminal_end:
            rollout.r = policy.value(last_obs, last_state)

        yield rollout

class A3C(object):
    def __init__(self, env, task):
        self.env  = env
        self.task = task

        # Seed the random number generator for reproducible initialization
        rng = np.random.RandomState(task)

        # Devices
        worker_device  = '/job:worker/task:{}/cpu:0'.format(task)
        replica_device = tf.train.replica_device_setter(
            1, worker_device=worker_device
            )

        # Shared network
        with tf.device(replica_device):
            with tf.variable_scope('global'):
                self.network = Policy(env.observation_space.shape,
                                      env.action_space.n, rng=rng)
                init = tf.constant_initializer(0, dtype=tf.int32)
                self.global_step = tf.get_variable('global_step',
                                                   [],
                                                   tf.int32,
                                                   initializer=init,
                                                   trainable=False)

        # Worker network
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.local_network = pi = Policy(env.observation_space.shape,
                                                 env.action_space.n, rng=rng)
                pi.global_step = self.global_step

            self.ac  = tf.placeholder(tf.float32, [None, env.action_space.n],
                                      name='ac')
            self.adv = tf.placeholder(tf.float32, [None], name='adv')
            self.r   = tf.placeholder(tf.float32, [None], name='r')

            # Action probabilities
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf     = tf.nn.softmax(pi.logits)

            # Policy loss
            pi_loss = tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv
            pi_loss = -tf.reduce_sum(pi_loss)

            # Value loss
            vf_loss = tf.square(pi.vf - self.r)
            vf_loss = 0.5 * tf.reduce_sum(vf_loss)

            # Entropy term
            entropy = -tf.reduce_sum(prob_tf * log_prob_tf)

            # Total loss
            lambda_v = 0.5
            lambda_e = 0.01
            self.loss = pi_loss + lambda_v * vf_loss - lambda_e * entropy

            # Runner
            self.runner = RunnerThread(env, pi)

            # Policy gradients
            grads = tf.gradients(self.loss, pi.variables)

            # Norms
            grad_norm = tf.global_norm(grads)
            var_norm  = tf.global_norm(pi.variables)

            # Add to summary
            bs = tf.to_float(tf.shape(pi.x)[0])
            tf.summary.scalar('model/policy_loss', pi_loss / bs)
            tf.summary.scalar('model/value_loss', vf_loss / bs)
            tf.summary.scalar('model/entropy', entropy / bs)
            tf.summary.scalar('model/grad_global_norm', grad_norm)
            tf.summary.scalar('model/var_global_norm', var_norm)
            tf.summary.image('observation', pi.x)
            self.summary_op = tf.summary.merge_all()

            # Copy weights from the parameter server to the local model
            sync_op = [v1.assign(v2)
                       for v1, v2 in zip(pi.variables, self.network.variables)]
            self.sync_op = tf.group(*sync_op)

            # Increment step
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # Learning rate
            learning_rate = 1e-4

            # Each worker gets its own optimizer
            optimizer  = tf.train.AdamOptimizer(learning_rate)
            grads, _   = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = zip(grads, self.network.variables)
            train_op   = optimizer.apply_gradients(grads_vars)

            self.train_op = tf.group(train_op, inc_step)
            self.summary_writer = None
            self.local_steps    = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.done:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        # Copy weights from shared to local
        sess.run(self.sync_op)

        # Get batch
        rollout = self.pull_batch_from_queue()
        batch   = process_rollout(rollout)

        self.local_steps += 1
        summary = (self.task == 0 and self.local_steps % 10 == 0)

        fetches = [self.train_op, self.global_step]
        if summary:
            fetches += [self.summary_op]
        feed_dict = {
            self.local_network.x: batch.obs,
            self.ac:  batch.a,
            self.adv: batch.adv,
            self.r:   batch.r,
            self.local_network.state_in: batch.state
            }
        fetched = sess.run(fetches, feed_dict)

        if summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[-1]),
                                            fetched[1])
            self.summary_writer.flush()
