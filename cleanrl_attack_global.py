import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import Monitor
import time
import torch
import random
import os
from wrappers import wrap_atari, wrap_deepmind

from attacks import attack
from g_re import GRe
from models import NoisyNet, QNetwork
from models import CnnDQN
from utils import Logger


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DQN agent')
    parser.add_argument('--gym-id', type=str, default="BoxingNoFrameskip-v4",
                        help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experiment')
    parser.add_argument('--restrict-actions', type=int, default=0,
                        help='restrict actions or not')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--load-checkpoint', type=str, default='frame_900000_reward90.2.pth',
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--dqn-type', type=str, default='noisynet')  # nature noisynet graddqn
    parser.add_argument('--print-frequency', type=int, default=100,
                        help="the frequency of saving")
    parser.add_argument('--max-episodes', type=int, default=10, help="max number of episodes")
    parser.add_argument('--max-frames-per-episode', type=int, default=500, help="max frames per episode")
    parser.add_argument('--sigma', type=float, default=0.01, help="standard deviation of Gaussian noise")
    parser.add_argument('--epsilon', type=float, default=0.01, help="attack epsilon")


    args = parser.parse_args()

    prefix = f'{args.gym_id}_{args.dqn_type}_global_nr-{args.max_episodes}_sigma-{args.sigma}_attack_eps-{args.epsilon}'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    train_log = os.path.join(prefix, 'test.log')
    logger = Logger(open(train_log, 'w'))

    experiment_name = f"{args.gym_id}__{int(time.time())}__{args.dqn_type}"
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    env = gym.make(args.gym_id)
    env = wrap_atari(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # records episode reward in `info['episode']['r']`
    if args.capture_video:
        env = Monitor(env, f'videos/{experiment_name}')
    env = wrap_deepmind(
        env,
        episode_life=False,
        clip_rewards=True,
        frame_stack=True,
        scale=False,
        restrict_actions=args.restrict_actions
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    logger.log(f'sigma = {args.sigma}')

    '''load model
    '''

    if args.dqn_type=='nature' :
        Network=QNetwork
        logger.log('testing with nature DQN')
    elif args.dqn_type=='noisynet':
        Network=NoisyNet
        logger.log('testing with NoisyNet')
    elif args.dqn_type=='graddqn':
        Network=QNetwork
        logger.log('testing with GradDQN')
    elif args.dqn_type=='adv':
        Network=QNetwork
        logger.log('testing with adv')
    elif args.dqn_type=='aug':
        Network=QNetwork
        logger.log('testing with aug')
    elif args.dqn_type=='cov':
        Network=QNetwork
        logger.log('testing with cov')
    elif args.dqn_type=='pgd':
        Network=QNetwork
        logger.log('testing with pgd')
    elif args.dqn_type=='radialrl':
        Network=CnnDQN
        logger.log('testing with radialrl')

    q_network = Network(env).to(device)
    if args.load_checkpoint:
        logger.log(f"Loading a policy - {args.load_checkpoint} ")
        if args.dqn_type=='graddqn' or args.dqn_type=='noisynet' or args.dqn_type=='radialrl':
            q_network.load_state_dict(torch.load(args.load_checkpoint))
        else:
            q_network.network.load_state_dict(torch.load(args.load_checkpoint))
    else:
        raise ValueError('Path {} not exists, please specify test model path.')

    '''initialize certification scheme
    '''

    g_re = GRe(logger.log, env.observation_space.shape, q_network, q_network.forward, args.sigma)

    obs = env.reset()
    episode_reward = 0
    episode_rewards = []

    num_episodes = 0
    this_episode_frame = 1

    if args.dqn_type == 'noisynet':
        q_network.sample()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        obs = np.array(obs)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        state /= 255
        state = attack(g_re, state, attack_config=None, epsilon=args.epsilon, network_type=args.dqn_type, smooth_type='global')

        if args.dqn_type == 'noisynet':
            q_network.sample()
            # state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        # logits = q_network(state)
        # action = torch.argmax(logits, dim=1).tolist()[0]
        action = g_re.forward(state)[0]

        next_obs, reward, done, info = env.step(action)
        episode_reward += reward

        if this_episode_frame == args.max_frames_per_episode:
            logger.log('maximum number of frames reached in this episode, reset environment!')
            done = True

        obs = next_obs
        if done:
            print(f'num_episodes = {num_episodes}, episode_reward = {episode_reward}')
            episode_rewards.append(episode_reward)
            # important to note that because `EpisodicLifeEnv` wrapper is applied,
            # the real episode reward is actually the sum of episode reward of 5 lives
            # which we record through `info['episode']['r']` provided by gym.wrappers.RecordEpisodeStatistics
            obs, episode_reward = env.reset(), 0

            num_episodes += 1
            this_episode_frame = 1
            if num_episodes == args.max_episodes:
                break

        else:
            this_episode_frame += 1

        if global_step% args.print_frequency == 0:
            logger.log(f'total frame:{global_step},avg last 10 episodes reward: {np.average(episode_rewards[:-11:-1])},'
                       f'avg episode reward: {np.average(episode_rewards)}')

    logger.log(f'mean reward:{np.mean(episode_rewards)},std: {np.std(episode_rewards)}')

    filename = f'{train_log[:-4]}_global-attacked-reward.pt'
    torch.save(episode_rewards, filename)
    logger.log(f'episode rewrads saved to {filename}')
