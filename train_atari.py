
# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import Logger,ReplayBuffer
from models import NoisyNet,QNetwork

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from wrappers import wrap_atari,wrap_deepmind
import sys



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="BoxingNoFrameskip-v4",
                        help='the id of the gym environment')
    parser.add_argument('--frame-stack', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Stack 4 last frames')
    parser.add_argument('--restrict-actions', type=int, default=0,
                        help='restrict actions or not')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--load-checkpoint', type=str, default='',
                        help='Where checkpoint file should be loaded from (usually results/checkpoint_restore.pth)')

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1.,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=80000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=4,
                        help="the frequency of training")
    parser.add_argument('--save-frequency', type=int, default=100000,
                        help="the frequency of saving")
    parser.add_argument('--dqn-type',type=str,default='nature')     #nature noisynet graddqn
    args = parser.parse_args()

prefix=f'{args.gym_id}_{args.dqn_type}'
if not os.path.exists(prefix):
    os.makedirs(prefix)
train_log=os.path.join(prefix,'train.log')
logger = Logger(open(train_log,'w'))


# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}__{args.dqn_type}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
               name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
env = wrap_atari(env)
env = gym.wrappers.RecordEpisodeStatistics(env)  # records episode reward in `info['episode']['r']`
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')
env = wrap_deepmind(
    env,
    clip_rewards=True,
    frame_stack=args.frame_stack,
    scale=False,
    restrict_actions=args.restrict_actions
)
a=env.unwrapped.get_action_meanings()[:env.action_space.n]

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# respect the default timelimit
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"



def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def grad_act(q_network,target_network,state,optimizer):
    state_adv = state
    state.requires_grad = True
    state.retain_grad()
    Beta = torch.distributions.beta.Beta(1.0, 1.0)

    q_values = q_network(state)
    policy = F.softmax(q_values, 1)
    _, best_action = q_values.max(1)
    _, w_action = q_values.min(1)

    target_values = q_network(state)
    max_values, _ = target_values.max(1)
    # policy=F.softmax(target_values,1)
    w_label = [0.0 for i in range(policy.size(1))]
    w_label[w_action.item()] = 1.0
    w_label = torch.tensor(w_label, device=device)

    J = -torch.sum(w_label * policy.squeeze(dim=0))

    # state.requires_grad=True
    # state.retain_grad()
    optimizer.zero_grad()
    J.backward()
    grad = state.grad.cuda()
    grad_dir = grad / torch.norm(grad, p=2)
    with torch.no_grad():
        for i in range(10):
            ni = Beta.sample(grad.shape).cuda()
            si = state - 0.01 * ni * grad_dir
            _, action_adv = q_network(si).max(1)
            value_adv = target_network(state)[:, action_adv].squeeze(dim=0)

            if value_adv.item() < max_values.item():
                max_values = value_adv
                state_adv = si

        q_values = q_network(state_adv)
        _, action = q_values.max(1)
    return action.item()

def mini_test(model, args, logger, num_episodes=10, max_frames_per_episode=30000):
    logger.log('start mini test')
    #training_config = config['training_config']
    #env_params = training_config['env_params']
    env_params={}
    env_params['clip_rewards'] = False
    env_params['frame_stack'] = args.frame_stack
    env_params['restrict_actions'] = args.restrict_actions
    #env_id = config['env_id']

    env = gym.make(args.gym_id)
    env = wrap_atari(env)
    env = wrap_deepmind(env, **env_params)
    all_rewards = []
    episode_reward = 0

    seed = random.randint(0, sys.maxsize)
    logger.log('reseting env with seed', seed)
    env.seed(seed)
    obs = env.reset()

    episode_idx = 1
    this_episode_frame = 1
    for frame_idx in range(1, num_episodes * max_frames_per_episode + 1):
        obs = np.array(obs)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        state /= 255
        logits = model(state)
        action = torch.argmax(logits, dim=1).tolist()[0]
        next_obs, reward, done, _ = env.step(action)

        # logger.log(action)
        obs = next_obs
        episode_reward += reward
        if this_episode_frame == max_frames_per_episode:
            logger.log('maximum number of frames reached in this episode, reset environment!')
            done = True

        if done:
            logger.log('reseting env with seed', seed)
            obs = env.reset()
            all_rewards.append(episode_reward)
            logger.log('episode {}/{} reward: {:6g}'.format(episode_idx, num_episodes, all_rewards[-1]))
            episode_reward = 0
            this_episode_frame = 1
            episode_idx += 1
            if episode_idx > num_episodes:
                break
        else:
            this_episode_frame += 1
    return np.mean(all_rewards)

use_ddqn=True

use_grad=False
if args.dqn_type=='nature' :
    Network=QNetwork
    logger.log('training with nature DQN')
elif args.dqn_type=='noisynet':
    Network=NoisyNet
    logger.log('training with NoisyNet')
elif args.dqn_type=='graddqn':
    Network=QNetwork
    logger.log('training with GradDQN')
    use_grad=True
else:
    print(f'We don\'t have {args.dqn_type} Qnet')
rb = ReplayBuffer(args.buffer_size)
q_network = Network(env).to(device)
target_network = Network(env).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

start_step=0
# training from checkpoint
if args.load_checkpoint:
    logger.log(f"Loading a policy - {args.load_checkpoint} ")
    checkpoint = torch.load(args.load_checkpoint)
    q_network.load_state_dict(
        checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step=checkpoint['frames']
    start_e=checkpoint['epsilon']
    logger.log(f'training from frames:{start_step},epsilon:{start_e}')

    if args.start_e>args.end_e:
        temp_fraction=args.exploration_fraction
        args.exploration_fraction=(start_e-args.end_e)*temp_fraction/(args.start_e-args.end_e)
        args.start_e=start_e

target_network.load_state_dict(q_network.state_dict())
#optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

best_test_reward = -float('inf')
# TRY NOT TO MODIFY: start the game
obs = env.reset()
episode_reward = 0
for global_step in range(start_step,args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
    obs = np.array(obs)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        state /= 255
        if args.dqn_type == 'noisynet':
            q_network.sample()
        if not use_grad:
            #state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            logits = q_network(state)
            action = torch.argmax(logits, dim=1).tolist()[0]
        else:
            action= grad_act(q_network,target_network,state,optimizer)

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, info = env.step(action)
    episode_reward += reward

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if 'episode' in info.keys():
        print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
        writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
        writer.add_scalar("charts/e psilon", epsilon, global_step)

    # ALGO LOGIC: training.
    rb.put((obs, action, reward, next_obs, done))
    if global_step > args.learning_starts+start_step and global_step % args.train_frequency == 0:
        if args.dqn_type=='noisynet':
            q_network.sample()
            target_network.sample()

        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)

        states = torch.from_numpy(s_obs).float().to(device)
        states /= 255
        next_states = torch.from_numpy(s_next_obses).float().to(device)
        next_states /= 255
        with torch.no_grad():
            if not use_ddqn:
                target_max = torch.max(target_network(next_states), dim=1)[0]
                td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (
                            1 - torch.Tensor(s_dones).to(device))
            else:
                _, max_next_action = q_network(next_states).max(1)
                target_max = target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
                td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (
                        1 - torch.Tensor(s_dones).to(device))
        old_val = q_network(states).gather(1,torch.LongTensor(s_actions).view(-1, 1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs
    if done:
        # important to note that because `EpisodicLifeEnv` wrapper is applied,
        # the real episode reward is actually the sum of episode reward of 5 lives
        # which we record through `info['episode']['r']` provided by gym.wrappers.RecordEpisodeStatistics
        obs, episode_reward = env.reset(), 0
    if global_step > args.learning_starts and global_step % args.save_frequency == 0:
        test_reward=mini_test(q_network,args,logger)
        logger.log('this test avg reward: {:6g}, step: {}'.format(test_reward,global_step))
        if test_reward >= best_test_reward:
            logger.log('new best reward {:6g} achieved, update checkpoint'.format(test_reward))
        best_test_reward = test_reward
        torch.save(q_network.state_dict(), f'{prefix}/frame_{global_step}_reward{round(test_reward, 1)}.pth')
        torch.save({
            'frames': global_step+1,
            'model_state_dict': q_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
        }, f'{prefix}/frame_{global_step}_reward{round(test_reward, 1)}_restore.pth')
env.close()
writer.close()