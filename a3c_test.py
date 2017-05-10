"""
Asynchronous advantage actor-critic (A3C) based on

    https://github.com/openai/universe-starter-agent

Original paper:

    Asynchronous methods for deep reinforcement learning.
    https://arxiv.org/abs/1602.01783

"""
import os
import numpy as np
import go_vncdriver # Must be imported before tensorflow
import tensorflow as tf
from gym.monitoring import video_recorder

from a3c_envs import create_env
from a3c_model import Policy

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('env-id', 'PongDeterministic-v3', "environment ID")
tf.app.flags.DEFINE_string('log-dir', '/tmp/pong', "log directory")
tf.app.flags.DEFINE_integer('seed', 0, "random number generator seed")
tf.app.flags.DEFINE_integer('num-episodes', 2, "number of episodes to run")
tf.app.flags.DEFINE_string('movie-path', '', "movie path")

class Agent(object):
    def __init__(self, env, sess):
        self.env  = env
        self.sess = sess

        # Model
        with tf.variable_scope('global'):
            self.policy = Policy(env.observation_space.shape,
                                 env.action_space.n)

        # Saver
        saver = tf.train.Saver()

        # Load model
        ckpt_dir = os.path.join(FLAGS.log_dir, 'train')
        ckpt     = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint available.")

    def run(self, num_episodes, video=None):
        env    = self.env
        sess   = self.sess
        policy = self.policy

        for n in range(num_episodes):
            obs = env.reset()
            if video is not None:
                video.capture_frame()
            state = policy.get_initial_state()

            length = 0
            while True:
                # Run policy for one step
                action, value, state = policy.act(obs, state)

                # Perform the action
                obs, reward, done, info = env.step(action.argmax())
                if video is not None:
                    video.capture_frame()

                # Add to episode
                length += 1

                # Done
                tag = 'wrapper_config.TimeLimit.max_episode_steps'
                max_length = env.spec.tags.get(tag)
                if done or length >= max_length:
                    break

#///////////////////////////////////////////////////////////////////////////////

def main(_):
    env = create_env(FLAGS.env_id, seed=FLAGS.seed)
    if FLAGS.movie_path:
        video = video_recorder.VideoRecorder(env=env,
                                             base_path=FLAGS.movie_path)
    else:
        video = None

    tf.set_random_seed(FLAGS.seed)
    with tf.Session() as sess:
        agent = Agent(env, sess)
        agent.run(FLAGS.num_episodes, video)

    if video is not None:
        print("Saving movie to {}.mp4".format(FLAGS.movie_path))
        video.close()

if __name__ == '__main__':
    tf.app.run()
