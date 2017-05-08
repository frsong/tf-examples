"""
Asynchronous advantage actor-critic based on

    https://github.com/openai/universe-starter-agent

Original paper:

    Asynchronous methods for deep reinforcement learning.
    https://arxiv.org/abs/1602.01783

"""
import os
import shlex
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num-workers', 1, "number of workers")
tf.app.flags.DEFINE_string('env-id', 'PongDeterministic-v3', "environment ID")
tf.app.flags.DEFINE_string('log-dir', '/tmp/pong', "log directory")
tf.app.flags.DEFINE_string('mode', 'tmux', "tmux, nohup, or child")
tf.app.flags.DEFINE_string('shell', 'bash', "shell")

def new_cmd(session, name, cmd, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join(shlex.quote(str(v)) for v in cmd)

    if FLAGS.mode == 'tmux':
        cmd = ('tmux send-keys -t {}:{} {} Enter'
               .format(session, name, shlex.quote(cmd)))
    elif FLAGS.mode == 'nohup':
        cmd = ('nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh'
               .format(shell, shlex.quote(cmd), FLAGS.log_dir, session, name,
                       FLAGS.log_dir))
    elif FLAGS.mode == 'child':
        cmd = ('{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh'
               .format(cmd, FLAGS.log_dir, session, name, FLAGS.log_dir))
    else:
        raise ValueError(FLAGS.mode)

    return name, cmd

def create_commands(session, shell='bash'):
    # For launching the TF workers
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable,  'a3c_worker.py',
        '--log-dir',     FLAGS.log_dir,
        '--env-id',      FLAGS.env_id,
        '--num-workers', str(FLAGS.num_workers)
        ]
    cmds_map = [new_cmd(session, 'ps', base_cmd + ['--job-name', 'ps'], shell)]
    for i in range(FLAGS.num_workers):
        cmd = base_cmd + ['--job-name', 'worker', '--task', str(i)]
        cmds_map.append(new_cmd(session, 'w-{}'.format(i), cmd, shell))
    if FLAGS.mode == 'tmux':
        cmds_map.append(new_cmd(session, 'htop', ['htop'], shell))
    windows, window_cmds = zip(*cmds_map)

    notes = []
    cmds  = [
        'mkdir -p {}'.format(FLAGS.log_dir),
        'echo {} {} > {}/cmd.sh'
        .format(sys.executable,
                ' '.join(shlex.quote(arg) for arg in sys.argv if arg != '-n'),
                FLAGS.log_dir)
        ]
    if FLAGS.mode == 'nohup' or FLAGS.mode == 'child':
        cmds.append('echo "#!/bin/sh" >{}/kill.sh'.format(FLAGS.log_dir))
        notes.append("Run `source {}/kill.sh` to kill the job"
                     .format(FLAGS.log_dir))
    if FLAGS.mode == 'tmux':
        notes.append("Use `tmux attach -t {}` to watch process output"
                     .format(session))
        notes.append("Use `tmux kill-session -t {}` to kill the job"
                     .format(session))
    else:
        notes.append("Use `tail -f {}/*.out` to watch process output"
                     .format(FLAGS.log_dir))

    if FLAGS.mode == 'tmux':
        cmds += [
            'kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1'
            .format(FLAGS.num_workers+12222),
            'tmux kill-session -t {}'.format(session),
            'tmux new-session -s {} -n {} -d {}'
            .format(session, windows[0], shell)
            ]
        for w in windows[1:]:
            cmds.append('tmux new-window -t {} -n {} {}'
                        .format(session, w, shell))
        cmds.append('sleep 1')
    cmds += window_cmds

    return cmds, notes

#///////////////////////////////////////////////////////////////////////////////

def main(_):
    cmds, notes = create_commands('a3c')
    print("Executing the following commands")
    print("--------------------------------")
    print("\n".join(cmds))
    print("")
    if FLAGS.mode == 'tmux':
        os.environ['TMUX'] = ''
    os.system('\n'.join(cmds))
    print("\n".join(notes))

if __name__ == '__main__':
    tf.app.run()
