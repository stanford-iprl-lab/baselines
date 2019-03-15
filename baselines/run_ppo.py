import json
import sys
import multiprocessing
import os.path as osp
import os
from datetime import datetime

import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
import baselines.ppo2.ppo2 as ppo2
from baselines import logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import pybulletgym
except ImportError:
    pybulletgym = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, ppo_args, network_args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    ppo_kwargs = get_learn_function_defaults('ppo', env_type)
    ppo_kwargs.update(vars(ppo_args))

    network_kwargs = vars(network_args)
    if extra_args:
        network_kwargs.update(vars(extra_args))

    lr = ppo_kwargs['lr']
    ppo_kwargs['lr'] = lambda f:f*lr
    cliprange = ppo_kwargs['cliprange']
    ppo_kwargs['cliprange'] = lambda f:f*cliprange

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.Logger.CURRENT.dir, "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        ppo_kwargs['network'] = args.network
    else:
        if ppo_kwargs.get('network') is None:
            ppo_kwargs['network'] = get_default_network(env_type)

    print('Training ppo on {}:{} with arguments \n{}'.format(env_type, env_id, ppo_kwargs))

    model = ppo2.learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **ppo_kwargs,
        **network_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = 'ppo'
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
       config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
       config.gpu_options.allow_growth = True
       get_session(config=config)

       flatten_dict_observations = alg not in {'her'}
       env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

       if env_type == 'mujoco':
           env = VecNormalize(env)

    return env


def get_env_type(env_id):
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def ppo_arg_parser():
    parser = arg_parser()
    parser.add_argument('--nsteps', type=int, default=2048, help='number of steps of the vectorized '
                                                                 'environment per update (i.e. batch'
                                                                 ' size is nsteps * nenv where nenv'
                                                                 ' is number of environment copies'
                                                                 ' simulated in parallel)')
    parser.add_argument('--ent-coef', '--ent_coef', type=float, default=0.0, help='entropy coefficient')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--vf-coef', '--vf_coef', type=float, default=0.5, help='value fn loss coefficient')
    parser.add_argument('--max-grad-norm', '--max_grad_norm', type=float, default=0.5, help='grad norm clipping scalar')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--log-interval', '--log_interval', type=int, default=10, help='logging interval')
    parser.add_argument('--nminibatches', type=int, default=4, help='number of training minibatches per update')
    parser.add_argument('--noptepochs', type=int, default=4, help='number of training epochs per update')
    parser.add_argument('--cliprange', type=float, default=0.2, help='clipping range, r schedule function [0,1] -> R+'
                                                                     ' where 1 is beginning of the training')
    parser.add_argument('--save-interval', '--save_interval', type=int, default=0, help='number of timesteps between saving events')
    parser.add_argument('--load-path', '--load_path', type=str, default=None, help='path to load model from')
    return parser


def network_arg_parser():
    parser = arg_parser()
    parser.add_argument('--value_network', '--value-network', type=str, default=None,
                        choices=[None, 'copy', 'shared'],
                        help='bool to decide if value network is to be used')
    parser.add_argument('--normalize_observations', '--normalize-observations', type=bool, default=False,
                        help='decide whether to normalize observations')
    parser.add_argument('--estimate_q', '--estimate-q', type=bool, default=False,
                        help='whether policy should estimate q or v')
    parser.add_argument('--num_layers', '--num-layers', type=int, default=2)
    parser.add_argument('--num_hidden', '--num-hidden', type=int, default=64)
    parser.add_argument('--layer_norm', '--layer-norm', type=bool, default=False)
    return parser


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--exp-name', '--exp_name', help='experiment name', type=str, default='ppo-exp')
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)',
                        choices=['mlp', 'cnn', 'lstm', 'cnn_lstm', 'conv_only'], default='mlp')
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--extra_import', help='Extra module to import to access external environments', type=str, default=None)
    return parser


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    args_parser = common_arg_parser()
    ppo_parser = ppo_arg_parser()
    network_parser = network_arg_parser()
    args, ppo_args = args_parser.parse_known_args(args)
    ppo_args, network_args = ppo_parser.parse_known_args(ppo_args)
    network_args, extra_args = network_parser.parse_known_args(network_args)
    extra_args = parse_cmdline_kwargs(extra_args)

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = osp.join(os.getenv('OPENAI_LOGDIR'), args.exp_name, dt)

    if args.extra_import is not None:
        import_module(args.extra_import)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=save_dir)
        with open(osp.join(logger.get_dir(), 'run.conf'), 'wt') as fh:
            print(datetime.now().isoformat(), file=fh)
            print(json.dumps(vars(args), indent=2), file=fh)
            print(json.dumps(vars(ppo_args), indent=2), file=fh)
            print(json.dumps(vars(network_args), indent=2), file=fh)
            if extra_args:
                print(json.dumps(vars(extra_args), indent=2), file=fh)
    else:
        logger.configure(format_strs=[])  # disables logging
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, ppo_args, network_args, extra_args)
    env.close()

    if rank == 0:
        save_path = osp.join(save_dir, '{}_{}'.format(args.env, args.num_timesteps))
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
