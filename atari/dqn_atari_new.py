# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
from gym import spaces
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import math
from gym.spaces import Discrete
from gym.wrappers import Monitor
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
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from eps_scheduler import EpsilonScheduler
from attacks import attack
from torch.nn import CrossEntropyLoss

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]

def get_logits_lower_bound(model, state, state_ub, state_lb, eps, C, beta):
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=state_lb, x_U=state_ub)
    bnd_state = BoundedTensor(state, ptb)
    pred = model.network(bnd_state, method_opt="forward")
    logits_ilb, _ = model.network.compute_bounds(C=C, IBP=True, method=None)
    if beta < 1e-5:
        logits_lb = logits_ilb
    else:
        logits_clb, _ = model.network.compute_bounds(IBP=False, C=C, method="backward", bound_upper=False)
        logits_lb = beta * logits_clb + (1-beta) * logits_ilb
    return logits_lb

def logits_margin(logits, y):
    comp_logits = logits - torch.zeros_like(logits).scatter(1, torch.unsqueeze(y, 1), 1e10)
    sec_logits, _ = torch.max(comp_logits, dim=1)
    margin = sec_logits - torch.gather(logits, 1, torch.unsqueeze(y, 1)).squeeze(1)
    margin = margin.sum()
    return margin


def compute_td_loss(current_model, target_model, batch_size, replay_buffer, bound_solver, natural_loss_fn, eps, sa, hinge, hinge_c, atk_method, atk_verbose, atk_eps, atk_niters, atk_img_min, atk_img_max, atk_random_start, kappa, max_norm, optimizer, gamma, robust, beta, gaussian_aug, aug_sigma):
        
    data = replay_buffer.sample(batch_size)
    state, next_state, action, reward, done = data.observations, data.next_observations, data.actions, data.rewards, data.dones

    optimizer.zero_grad()

    state = state.to(torch.float)
    next_state = next_state.to(torch.float)
    
    state /= 255
    next_state /= 255
    state_max = 1.0
    state_min = 0.0
    
    if gaussian_aug:
        delta = torch.randn_like(state) * aug_sigma
        state += delta
        next_state += delta

    if robust and bound_solver != 'pgd':
        cur_q_logits = current_model(state, method_opt="forward")
        tgt_next_q_logits = target_model(next_state, method_opt="forward")
    else:
        cur_q_logits = current_model(state)
        tgt_next_q_logits = target_model(next_state)
    
#     print(cur_q_logits.shape)
#     print(action.shape)
#     print(action.unsqueeze(1).shape)
#     cur_q_value = cur_q_logits.gather(1, action.unsqueeze(1)).squeeze(1)
    cur_q_value = cur_q_logits.gather(1, action).squeeze(1)

    tgt_next_q_value = tgt_next_q_logits.max(1)[0]
    expected_q_value = reward.squeeze(1) + gamma * tgt_next_q_value * (1 - done.squeeze(1))
#     print(cur_q_value.shape)
#     print(tgt_next_q_value.shape)
#     print(done.shape)
#     print(reward.shape)
#     print(expected_q_value.shape)
    '''
    # Merge two states into one batch
    state = state.to(torch.float)
    if dtype in UINTS:
        state /= 255
    state_and_next_state = torch.cat((state, next_state), 0)
    logits = current_model(state_and_next_state)
    cur_q_logits = logits[:state.size(0)]
    cur_next_q_logits = logits[state.size(0):]
    tgt_next_q_value  = tgt_next_q_logits.gather(1, torch.max(cur_next_q_logits, 1)[1].unsqueeze(1)).squeeze(1)
    '''

    if natural_loss_fn == 'huber':
#         loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        loss_fn = nn.MSELoss()
        loss = loss_fn(cur_q_value, expected_q_value.detach())
    else:
        loss  = (cur_q_value - expected_q_value.detach()).pow(2)
        loss = loss.mean()

    batch_cur_q_value = torch.mean(cur_q_value)
    batch_exp_q_value = torch.mean(expected_q_value)
#     loss = loss.mean()
#     td_loss = loss.clone()

    if robust:
        if eps < np.finfo(np.float32).tiny:
            reg_loss = torch.zeros(state.size(0))
            if USE_CUDA:
                reg_loss = reg_loss.cuda()
            if bound_solver == 'pgd':
                labels = torch.argmax(cur_q_logits, dim=1).clone().detach()
                adv_margin = ori_margin = logits_margin(current_model.forward(state), labels)
                optimizer.zero_grad()
        else:
            if bound_solver != 'pgd':
                pred = cur_q_logits
                labels = torch.argmax(pred, dim=1).clone().detach()
                c = torch.eye(current_model.num_actions).type_as(state)[labels].unsqueeze(1) - torch.eye(current_model.num_actions).type_as(state).unsqueeze(0)
                I = (~(labels.data.unsqueeze(1) == torch.arange(current_model.num_actions).type_as(labels.data).unsqueeze(0)))
                c = (c[I].view(state.size(0), current_model.num_actions-1, current_model.num_actions))
                sa_labels = sa[labels]
                lb_s = torch.zeros(state.size(0), current_model.num_actions)
                if USE_CUDA:
                    labels = labels.cuda()
                    c = c.cuda()
                    sa_labels = sa_labels.cuda()
                    lb_s = lb_s.cuda()

                eps_v = eps
                state_max = 1.0
                state_min = 0.0
                state_ub = torch.clamp(state.permute((0, 3, 1, 2)) + eps_v, max=state_max)
                state_lb = torch.clamp(state.permute((0, 3, 1, 2)) - eps_v, min=state_min)

                lb = get_logits_lower_bound(current_model, state.permute((0, 3, 1, 2)), state_ub, state_lb, eps_v, c, beta)

                if hinge:
                   reg_loss, _ = torch.min(lb, dim=1)
                   reg_loss = torch.clamp(reg_loss, max=hinge_c)
                   reg_loss = - reg_loss
                else:
                    lb = lb_s.scatter(1, sa_labels, lb)
                    reg_loss = CrossEntropyLoss()(-lb, labels)
            else:
                labels = torch.argmax(cur_q_logits, dim=1).clone().detach()
                adv_state = attack(current_model, state, atk_method, atk_verbose, atk_eps, atk_niters, atk_img_min, atk_img_max, atk_random_start, logits_margin)
                optimizer.zero_grad()
                adv_margin = logits_margin(current_model.forward(adv_state), labels)
                ori_margin = logits_margin(current_model.forward(state), labels)
                reg_loss = torch.clamp(adv_margin, min=-hinge_c)

        reg_loss = reg_loss.mean()
        loss += kappa * reg_loss

    loss.backward()

    # Gradient clipping.
#     grad_norm = 0.0
#     if max_norm > 0:
#         parameters = current_model.parameters()
#         for p in parameters:
#             grad_norm += p.grad.data.norm(2).item() ** 2
#         grad_norm = np.sqrt(grad_norm)
#         clip_coef = max_norm / (grad_norm + 1e-6)
#         if clip_coef < 1:
#             for p in parameters:
#                 p.grad.data.mul_(clip_coef)
    nn.utils.clip_grad_norm_(list(current_model.parameters()), max_norm)

    # update weights
    optimizer.step()

    return loss

def parse_args():
    # Common arguments
    # fmt: off
    parser = argparse.ArgumentParser(description='PPO agent')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--act-num', type=int, default=0,
        help='action size')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=1000000,
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
    
    parser.add_argument('--robust', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, model will be trained in robust way')
    parser.add_argument('--bound-solver', type=str, default="cov",
        help="bound solver type")
    
    parser.add_argument('--schedule-start', type=int, default=1500000,
        help="timestep to start epsilon scheduler")
    parser.add_argument('--schedule-length', type=int, default=4000000,
        help="length to schedule epsilon")
    parser.add_argument('--start-epsilon', type=float, default=0.0,
        help="epsilon start from")
    parser.add_argument('--epsilon', type=float, default=0.00392,
        help="epsilon end with")
    parser.add_argument('--schedule-type', type=str, default="smoothed",
        help="epsilon schedule type")
    
    parser.add_argument('--convex-start-beta', type=float, default=1.0,
        help="beta start with")
    parser.add_argument('--convex-final-beta', type=float, default=0.0,
        help="beta end with")
    
    parser.add_argument('--kappa', type=float, default=0.005,
        help="kappa")
    parser.add_argument('--hinge', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, use hinge loss')
    parser.add_argument('--hinge-c', type=int, default=1,
        help="hinge value")
    
    parser.add_argument('--natural-loss-fn', type=str, default="huber",
        help="loss function to compute td loss")
    
    parser.add_argument('--atk-method', type=str, default="pgd",
        help="method to attack")
    parser.add_argument('--atk-verbose', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, use verbose')
    parser.add_argument('--atk-eps', type=float, default=0.00392,
        help="attack epsilon")
    parser.add_argument('--atk-niters', type=int, default=5,
        help="number of attack iterations")
    parser.add_argument('--atk-img-min', type=float, default=0.0,
        help="image pixel min value")
    parser.add_argument('--atk-img-max', type=float, default=1.0,
        help="image pixel max value")
    parser.add_argument('--atk-random-start', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, attach with random start')
    
    parser.add_argument('--adv-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, use adv train')
    parser.add_argument('--adv-ratio', type=float, default=0.5,
        help="adv training ratio")
    
    parser.add_argument('--gaussian-aug', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, use gaussian augmentation train')
    parser.add_argument('--aug-sigma', type=float, default=0.005,
        help="gaussian augmentation sigma")
    
    
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    # fmt: on
    return args

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
        
def make_env(gym_id, seed, idx, restrict_actions=0):
    def thunk():
        env = gym.make(gym_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.capture_video:
            if idx == 0:
                env = Monitor(env, f"videos/{experiment_name}")
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


class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class QNetwork(nn.Module):
    def __init__(self, env, frames=4, robust=False):
        super(QNetwork, self).__init__()
        self.num_actions = env.action_space.n
        self.robust = robust
        self.env = env
        self.network = nn.Sequential(
#             Scale(1 / 255),
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            Linear0(512, env.action_space.n),
        )
        
        if self.robust:
            dummy_input = torch.empty_like(torch.randn((1,) + (env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1])))
            self.network = BoundedModule(self.network, dummy_input, device="cuda" if USE_CUDA else "cpu")    

    def forward(self, x, method_opt="forward"):
        return self.network(x.permute((0, 3, 1, 2)).contiguous())
    
    def act(self, state, epsilon=0):
        #state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
        if self.robust:
            q_value = self.forward(state, method_opt='forward')
        else:
            q_value = self.forward(state)
        action  = q_value.max(1)[1].data.cpu().numpy()
        mask = np.random.choice(np.arange(0, 2), p=[1-epsilon, epsilon])
        action = (1-mask) * action + mask * np.random.randint(self.env.action_space.n, size=state.size()[0])
        return action

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )
    if args.prod_mode:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(f"/tmp/{experiment_name}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = VecFrameStack(
        DummyVecEnv([make_env(args.gym_id, args.seed, 0, args.act_num)]),
        4,
    )
    assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ALGO LOGIC: initialize agent here:
    rb = ReplayBuffer(args.buffer_size, envs.observation_space, envs.action_space, device=device, optimize_memory_usage=True)
    q_network = QNetwork(envs, 4, args.robust).to(device)
    target_network = QNetwork(envs, 4, args.robust).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    
    if args.robust or args.adv_train:
        schedule_start = args.schedule_start
        schedule_length = args.schedule_length
        starting_epsilon=  args.start_epsilon
        end_epsilon = args.epsilon
        epsilon_scheduler = EpsilonScheduler(args.schedule_type, schedule_start, schedule_start+schedule_length-1, starting_epsilon, end_epsilon, 1)
        max_eps = end_epsilon
    
    if args.adv_train:
        attack_count = 0
        suc_count = 0
    if args.robust and args.bound_solver == 'pgd':
        ori_margin, adv_margin = np.nan, np.nan    
    
    sa = None
    kappa = None
    hinge = False
    if args.robust:
        print('using convex relaxation certified classification loss as a regularization!')
        kappa = args.kappa
        reg_losses = []
        sa = np.zeros((envs.action_space.n, envs.action_space.n - 1), dtype = np.int32)
        for i in range(sa.shape[0]):
            for j in range(sa.shape[1]):
                if j < i:
                    sa[i][j] = j
                else:
                    sa[i][j] = j + 1
        sa = torch.LongTensor(sa)
        hinge = args.hinge
        print('using hinge loss (default is cross entropy): ', hinge)

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        eps = 0
        if args.adv_train or args.robust:
#         if adv_train or args.robust:
            eps = epsilon_scheduler.get_eps(global_step, 0)
            
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if args.adv_train and eps != np.nan and eps >= np.finfo(np.float32).tiny:
#             ori_state_tensor = torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0).cuda().to(torch.float32)
            ori_state_tensor = torch.Tensor(obs).to(device)
#             if dtype in UINTS:
#                 ori_state_tensor /= 255
            ori_state_tensor /= 255
            args.atk_eps = eps
            if random.random() < args.adv_ratio:
                attack_count += 1
                state_tensor = attack(q_network, ori_state_tensor, args.atk_method, args.atk_verbose, args.atk_eps, args.atk_niters, args.atk_img_min, args.atk_img_max, args.atk_random_start, logits_margin)
                if q_network.act(state_tensor)[0] != q_network.act(ori_state_tensor)[0]:
                    suc_count += 1
            else:
                state_tensor = ori_state_tensor
            action = q_network.act(state_tensor, epsilon)[0]
#             action = current_model.act(state_tensor, act_epsilon)[0]
        else:
            with torch.no_grad():
                state_tensor = torch.Tensor(obs).to(device)
#                 if dtype in UINTS:
#                     state_tensor /= 255
                state_tensor /= 255
                ori_state_tensor = torch.clone(state_tensor)
                action = q_network.act(state_tensor, epsilon)[0]
#                 action = current_model.act(state_tensor, act_epsilon)[0]
                
#         action = q_network.act(torch.Tensor(obs).to(device), epsilon)[0]
#         if random.random() < epsilon:
#             actions = [envs.action_space.sample()]
#         else:
#             logits = q_network.forward(torch.Tensor(obs).to(device))
#             actions = torch.argmax(logits, dim=1).cpu().numpy()
        
        next_obs, rewards, dones, infos = envs.step([action.tolist()])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info["episode"]["r"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, [action.tolist()], rewards, dones)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
#             data = rb.sample(args.batch_size)
#             with torch.no_grad():
#                 target_max, _ = target_network.forward(data.next_observations).max(dim=1)
#                 td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
#             old_val = q_network.forward(data.observations).gather(1, data.actions).squeeze()
#             loss = loss_fn(td_target, old_val)
            if args.robust: 
                convex_final_beta = args.convex_final_beta
                convex_start_beta = args.convex_start_beta
                beta = (max_eps - eps * (1.0 - convex_final_beta)) / max_eps * convex_start_beta
            else:
                beta = 0
                
            loss = compute_td_loss(q_network, target_network, args.batch_size, rb, args.bound_solver, args.natural_loss_fn, eps, sa, hinge, args.hinge_c, args.atk_method, args.atk_verbose, args.atk_eps, args.atk_niters, args.atk_img_min, args.atk_img_max, args.atk_random_start, kappa, args.max_grad_norm, optimizer, args.gamma, args.robust, beta, args.gaussian_aug, args.aug_sigma)
                

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)

#             # optimize the model
#             optimizer.zero_grad()
#             loss.backward()
#             nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
#             optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
                
            if global_step % 100000 == 0:
                if args.robust:
                    torch.save(q_network.network.state_dict(), '{}_{}/frame_{}.pth'.format(args.gym_id, args.bound_solver, global_step))
                else: 
                    if args.adv_train:
                        torch.save(q_network.network.state_dict(), '{}_adv/frame_{}.pth'.format(args.gym_id, global_step))
                    elif args.gaussian_aug:
                        torch.save(q_network.network.state_dict(), '{}_aug/frame_{}.pth'.format(args.gym_id, global_step))
                    else:
                        torch.save(q_network.network.state_dict(), '{}_nat/frame_{}.pth'.format(args.gym_id, global_step))
                
    
    if args.robust:
        torch.save(q_network.network.state_dict(), '{}_{}/frame_{}.pth'.format(args.gym_id, args.bound_solver, global_step))
    else: 
        if args.adv_train:
            torch.save(q_network.network.state_dict(), '{}_adv/frame_{}.pth'.format(args.gym_id, global_step))
        elif args.gaussian_aug:
            torch.save(q_network.network.state_dict(), '{}_aug/frame_{}.pth'.format(args.gym_id, global_step))
        else:
            torch.save(q_network.network.state_dict(), '{}_nat/frame_{}.pth'.format(args.gym_id, global_step))
    envs.close()
    writer.close()
