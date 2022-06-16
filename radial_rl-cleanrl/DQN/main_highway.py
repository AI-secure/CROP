from __future__ import print_function, division
import os
import argparse
import torch
# from environment import atari_env
from utils import read_config
from model import CnnDQN, MLP_QNetwork
from train_highway import train
#from gym.configuration import undo_logger_setup
import time

import gym
import numpy as np
from gym import spaces
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
from gym.wrappers import TimeLimit, Monitor
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import highway_env

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.000125,
    metavar='LR',
    help='learning rate (default: 0.000125)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='S',
    help='random seed (default: None)')
parser.add_argument(
    '--total-frames',
    type=int,
    default=6000000,
    metavar='TS',
    help='How many frames to train with')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='PongNoFrameskip-v4',
    metavar='ENV',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')

parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='optimizer to use, one of {Adam, RMSprop}')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=False,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--worse-bound',
    default=True,
    help='if this is selected worst case loss uses bound that is further away from mean')

parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--kappa-end',
    type=float,
    default=0.5,
    metavar='SR',
    help='final value of the variable controlling importance of standard loss (default: 0.5)')
parser.add_argument('--robust',
                   dest='robust',
                   action='store_true',
                   help='train the model to be verifiably robust')
parser.add_argument(
    '--load-path',
    type=str,
    default=None,
    help='Path to load a model from. By default starts training a new model')

parser.add_argument(
    '--attack-epsilon-end',
    type=float,
    default=1/255,
    metavar='EPS',
    help='max size of perturbation trained on')
parser.add_argument(
    '--attack-epsilon-schedule',
    type=int,
    default=3000000,
    help='The frame by which to reach final perturbation')
parser.add_argument(
    '--exp-epsilon-end',
    type=float,
    default=0.01,
    help='for epsilon-greedy exploration')
parser.add_argument(
    '--exp-epsilon-decay',
    type=int,
    default=500000,
    help='controls linear decay of exploration epsilon')
parser.add_argument(
    '--replay-initial',
    type=int,
    default=50000,
    help='How many frames of experience to collect before starting to learn')
parser.add_argument(
    '--batch-size',
    type=int,
    default=128,
    help='Batch size for updating agent')
parser.add_argument(
    '--updates-per-frame',
    type=int,
    default=32,
    help='How many gradient updates per new frame')
parser.add_argument(
    '--buffer-size',
    type=int,
    default=200000,
    help='How frames to store in replay buffer')
parser.add_argument(
    '--num-act',
    type=int,
    default=0,
    help='Size of action set')


parser.set_defaults(robust=False)

class RestrictedActionEnv(gym.Wrapper):
    def __init__(self, env, num_action):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        # Action mask is a hex number, where 0 bit masks the action
        if isinstance(num_action, bool):
            # 6 basic actions
            num_action = 6
        else:
            # specify the number of actions
            self.num_action = num_action
        print('Using the first {} actions only.'.format(num_action))
        self.action_space = spaces.Discrete(num_action)

def make_env(gym_id, seed, idx, restrict_actions):
    def thunk():
        env = gym.make(gym_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if args.capture_video:
        #     if idx == 0:
        #         env = Monitor(env, f"videos/{experiment_name}")
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if restrict_actions:
            env = RestrictedActionEnv(env, restrict_actions)
        env = WarpFrame(env, width=84, height=84)
        env = ClipRewardEnv(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

class ProcessObsInputEnv(gym.ObservationWrapper):
    """
    This wrapper handles inputs from `Discrete` and `Box` observation space.
    If the `env.observation_space` is of `Discrete` type, 
    it returns the one-hot encoding of the state
    """
    def __init__(self, env):
        super().__init__(env)
        self.n = None
        if isinstance(self.env.observation_space, Discrete):
            self.n = self.env.observation_space.n
            self.observation_space = Box(0, 1, (self.n,))

    def observation(self, obs):
        if self.n:
            return one_hot(np.array(obs), self.n)
        return obs

if __name__ == '__main__':
    args = parser.parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
        if args.gpu_id>=0:
            torch.cuda.manual_seed(args.seed)
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]

    # env = atari_env(args.env, env_conf, args)
#     env = VecFrameStack(
#         DummyVecEnv([make_env(args.env, args.seed, 0, args.num_act)]),
#         4,
#     )
    env = ProcessObsInputEnv(gym.make(args.env))
    assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
    
    # curr_model = CnnDQN(env.observation_space.shape[0], env.action_space)
#     curr_model = CnnDQN(env)
    curr_model = MLP_QNetwork(env)
    
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    
    if args.load_path:
        saved_state = torch.load(
            args.load_path,
            map_location=lambda storage, loc: storage)
        curr_model.network.load_state_dict(saved_state)
        
    # target_model = CnnDQN(env.observation_space.shape[0], env.action_space)
#     target_model = CnnDQN(env)
    target_model = MLP_QNetwork(env)
    target_model.load_state_dict(curr_model.state_dict())
    if args.gpu_id >= 0:
        with torch.cuda.device(args.gpu_id):
            curr_model.cuda()
            target_model.cuda()
            
    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(curr_model.parameters(), lr=args.lr, momentum=0.95, alpha=0.95, eps=1e-2)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(curr_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    
    train(curr_model, target_model, env, optimizer, args)
        
