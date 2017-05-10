"""
Asynchronous advantage actor-critic based on

    https://github.com/openai/universe-starter-agent

Original paper:

    Asynchronous methods for deep reinforcement learning.
    https://arxiv.org/abs/1602.01783

"""
import logging
import os
import signal
import sys
import time
import go_vncdriver # Must be imported before tensorflow
import tensorflow as tf

from a3c_a3c  import A3C
from a3c_envs import create_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('task', 0, "task index")
tf.app.flags.DEFINE_string('job-name', 'worker', "worker or ps")
tf.app.flags.DEFINE_integer('num-workers', 1, "number of workers")
tf.app.flags.DEFINE_string('log-dir', '/tmp/pong', "log directory")
tf.app.flags.DEFINE_string('env-id', 'PongDeterministic-v3', "environment ID")

# Custom saver that disables the `write_meta_graph` argument.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix='meta', write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step,
                                    latest_filename, meta_graph_suffix, False)

def run(server, seed):
    # Create environment
    env = create_env(FLAGS.env_id, seed)

    # Seed TF random number generator
    tf.set_random_seed(seed)

    # A3C
    a3c = A3C(env, FLAGS.task)

    # Don't save variables that start with 'local'
    variables_to_save = [v for v in tf.global_variables()
                         if not v.name.startswith('local')]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()

    # Saver
    saver = FastSaver(variables_to_save)

    # Print list of variables
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    logger.info("Trainable vars:")
    for v in variables:
        logger.info("  %s %s", v.name, v.get_shape())

    # Log directory
    log_dir = os.path.join(FLAGS.log_dir, 'train')

    # For TensorBoard
    worker_log_dir = log_dir + '_{}'.format(FLAGS.task)
    summary_writer = tf.summary.FileWriter(worker_log_dir)
    logger.info("Events directory: %s", worker_log_dir)

    def init_fn(sess):
        logger.info("Initializing all parameters.")
        sess.run(init_all_op)

    sv = tf.train.Supervisor(
        is_chief=(FLAGS.task == 0),
        logdir=log_dir,
        saver=saver,
        summary_op=None,
        init_op=init_op,
        init_fn=init_fn,
        summary_writer=summary_writer,
        ready_op=tf.report_uninitialized_variables(variables_to_save),
        global_step=a3c.global_step,
        save_model_secs=30,
        save_summaries_secs=30
        )

    filters = ['/job:ps', '/job:worker/task:{}/cpu:0'.format(FLAGS.task)]
    config  = tf.ConfigProto(device_filters=filters)
    with sv.managed_session(server.target, config=config) as sess:
        # Sync parameters
        sess.run(a3c.sync_op)

        a3c.start(sess, summary_writer)
        global_step = sess.run(a3c.global_step)
        logger.info("Starting training at step = %d", global_step)
        while not sv.should_stop() and global_step < 100000000:
            a3c.process(sess)
            global_step = sess.run(a3c.global_step)

    # Ask for all services to stop
    sv.stop()
    logger.info("Reached %s steps. worker stopped.", global_step)

def cluster_spec(num_workers, num_ps):
    cluster = {}
    port    = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers

    return cluster

def main(_):
    spec    = cluster_spec(FLAGS.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn("Received signal %s: exiting", signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP,  shutdown)
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if FLAGS.job_name == 'worker':
        server = tf.train.Server(
            cluster,
            job_name='worker',
            task_index=FLAGS.task,
            config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=2)
            )
        run(server, FLAGS.task)
    else:
        server = tf.train.Server(
            cluster,
            job_name='ps',
            task_index=FLAGS.task,
            config=tf.ConfigProto(device_filters=['/job:ps'])
            )
        while True:
            time.sleep(1000)

#///////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    tf.app.run()
