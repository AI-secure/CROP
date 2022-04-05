import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
import copy
from dataclasses import dataclass, field
from scipy.stats import norm
import numpy as np
from queue import PriorityQueue
import random
import time
from tqdm import tqdm

import torch

from lo_act import LoAct

UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]

global_id = 0

class Elem(object):
    def __init__(self, env, s, a, rad, re, no, node_id):
        global global_id
        self.gid = global_id
        global_id += 1

        self.env = env
        self.s = s
        self.a = a
        self.rad = rad
        self.re = re
        self.no = no
        self.node_id = node_id

    def __lt__(self, other):
        return self.rad < other.rad

def to_str(d):
    return ",".join("{}: {}".format(*i) for i in d.items())

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: object = field()

class LoRe:
    def __init__(self, log_func, input_shape, model, forward_func, 
                 m=10000, sigma=0.01, v_lo=0, v_hi=1, conf_bound=0.05, max_frames_per_episode=200):
        self.log_func = log_func
        self.model = model
        self.forward_func = forward_func
        self.m = m
        self.sigma = sigma
        self.v_lo = v_lo
        self.v_hi = v_hi
        self.delta = ((self.v_hi - self.v_lo) * np.sqrt(np.log(1 / conf_bound) / (2 * self.m)))
        self.log_func(f'v_lo = {v_lo}, v_hi = {v_hi}, delta = {self.delta}')

        self.max_frames_per_episode = max_frames_per_episode

        self.lo_act = LoAct(log_func, input_shape, model, forward_func, m, sigma, v_lo, v_hi, conf_bound)

        self.log_func(f'start generating noise list!')
        t = time.time()
        self.noise_list = [ torch.FloatTensor(*input_shape).normal_(0, self.sigma).cuda()
                        for _ in tqdm(range(self.m)) ]
        self.log_func(f'generating {self.m} noise done using {time.time() - t} seconds!')

        # initialize
        self.re_min = 1e100
        self.certify_map = {}
        self.state_set = [set() for _ in range(self.max_frames_per_episode+1)]
        self.state_dict = dict()

        self.node_id = 0

        self.p_que = PriorityQueue()
        self.p_que.put(PrioritizedItem(1e100, Elem(None, None, None, 1e100, 0, -1, self.node_id)))

        self.edges = []


    def take_action(self, env, state, rad_lim, re_cur, no, node_id):

        if type(state) != np.ndarray:
            state = np.array(state)

        # check whether state has been visited and determine whether needed to put in queue
        state_bytes = state.tobytes()
        if state_bytes in self.state_set[no]:
            self.log_func(f'########################################### duplicated states encountered, state_set size at no={no} is {len(self.state_set[no])}')
            vis = True
        else:
            self.state_set[no].add(state_bytes)
            vis = False

        # state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
        if state.dtype in UINTS:
            state_tensor /= 255

        a_star, tilde_q = self.lo_act.forward(state_tensor, return_q=True, cert=False)

        a_star = a_star[0]
        tilde_q = tilde_q[0]

        assert(torch.argmax(tilde_q) == a_star)

        a_list = []

        snapshot = env.ale.cloneState()
        for a in range(env.action_space.n):
            env.ale.restoreState(snapshot)

            next_state, reward, done, _ = env.step(a)

            if done:
                continue

            # reward shaping for Pong
            reward = max(reward, 0)

            val_1 = tilde_q[a_star].cpu()
            val_2 = tilde_q[a].cpu()

            if val_1 - self.delta >= val_2 + self.delta:
                rad = self.sigma / 2 * (
                                    norm.ppf((val_1 - self.delta - self.v_lo) / (self.v_hi - self.v_lo)) - 
                                    norm.ppf((val_2 + self.delta - self.v_lo) / (self.v_hi - self.v_lo))
                                    )
                self.log_func(f'certified: radius = {rad}')
            else:
                rad = 0     # cannot certify
                self.log_func(f'cannot certify: val_1 = {val_1}, val_2 = {val_2}, delta = {self.delta}, radius = 0')

            if np.isnan(rad):
                self.log_func(f'a_star: {tilde_q[a_star]}, a: {tilde_q[a]}, v_lo: {self.v_lo}, v_hi: {self.v_hi}')
                raise NotImplementedError

            if rad <= rad_lim:
                a_list.append(a)
            elif not vis:
                env.ale.restoreState(snapshot)  # revert back for storing

                elem = Elem(copy.deepcopy(env), state, a, rad, re_cur, no, node_id)
                self.p_que.put(PrioritizedItem(elem.rad, elem))
                self.log_func(f'appending to queue rad={elem.rad}')

        return a_list

    def update_dict_1_level(self, dic, k, v):
        dic[k] = v if k not in dic else min(dic[k], v)

    def update_dict_2_level(self, state_bytes, new_dict, notice=False):
        if state_bytes not in self.state_dict:
            self.state_dict[state_bytes] = new_dict
        else:
            for k, v in new_dict.items():
                if notice: 
                    assert k not in self.state_dict[state_bytes], f'k = {k} in self.state_dict[state_bytes], previous = {to_str(self.state_dict[state_bytes])}'
                self.update_dict_1_level(self.state_dict[state_bytes], k, v)


    def expand(self, env, state, rad_lim=0, re_cur=0, no=0, node_id=0):
        # self.log_func(f'no = {no}, re_cur = {re_cur}, rad_lim = {rad_lim}, sizeq = {len(p_que.queue)}, map = {len(certify_map)}')
        if type(state) != np.ndarray:
            state = np.array(state)
        state_bytes = state.tobytes()

        # pruning
        if re_cur >= self.re_min:
            self.update_dict_2_level(state_bytes, {self.max_frames_per_episode-no: 0})
            self.log_func(f'************************************************ pruning at no={no}')
            return self.state_dict[state_bytes]

        if no >= self.max_frames_per_episode:
            self.update_dict_2_level(state_bytes, {self.max_frames_per_episode-no: 0})
            self.re_min = min(self.re_min, re_cur)
            self.log_func(f'================================================ run to the end with re={re_cur}, updated re_min={self.re_min}')
            return self.state_dict[state_bytes]

        snapshot = env.ale.cloneState()

        # if have visited the state
        if state_bytes in self.state_dict:
            # if have visited the state at the same time step
            if self.max_frames_per_episode - no in self.state_dict[state_bytes]:
                re_all = re_cur+self.state_dict[state_bytes][self.max_frames_per_episode-no]
                self.re_min = min(self.re_min, re_all)
                self.log_func(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% expanding duplicated states with same length={self.max_frames_per_episode-no} at no={no}, achieve re_all={re_all}, updated re_min={self.re_min}')
                return self.state_dict[state_bytes]
            else:
                self.log_func(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% expanding duplicated states without length={self.max_frames_per_episode-no} at no={no}')
        self.log_func(f'++++++++++++++++++++++++++++++++++++++++++++++++ state dict size is {len(self.state_dict)}')

        # continue expanding
        a_list = self.take_action(env, state, rad_lim, re_cur, no, node_id)

        if not len(a_list):
            self.update_dict_2_level(state_bytes, {self.max_frames_per_episode-no: 0})
            self.re_min = min(self.re_min, re_cur)
            self.log_func(f'################################################ no action at no={no}, achieve re={re_cur}, updated re_min={self.re_min}!')
            return self.state_dict[state_bytes]

        if state_bytes in self.state_dict:
            self.log_func(f'at no={no}, before taking action, self.state_dict[state_bytes] = {to_str(self.state_dict[state_bytes])}')

        cur_dict = dict()

        for a in a_list:
            env.ale.restoreState(snapshot)

            next_state, reward, done, _ = env.step(a)

            self.node_id += 1
            self.edges.append((node_id, self.node_id, a, reward, rad_lim))

            if type(next_state) != np.ndarray:
                next_state = np.array(next_state)

            assert not done

            # reward shaping for Pong
            reward = max(reward, 0)

            self.log_func(f'no = {no+1}, re_cur = {re_cur+reward}, rad_lim = {rad_lim}, sizeq = {len(self.p_que.queue)}, map = {len(self.certify_map)}, a = {a} in a_list = {a_list}')

            next_dict = self.expand(env, next_state, rad_lim, re_cur+reward, no+1, self.node_id)
            self.log_func(f'action = {a}, next_dict = {to_str(next_dict)}')

            for k, v in next_dict.items():
                self.update_dict_1_level(cur_dict, k+1, reward+v)

        self.log_func(f'at no={no}, after exploring all actions, cur_dict = {to_str(cur_dict)}')

        self.update_dict_2_level(state_bytes, cur_dict, notice=False)
        self.log_func(f'at no={no}, after merging dict, self.state_dict[state_bytes] = {to_str(self.state_dict[state_bytes])}')

        return self.state_dict[state_bytes]


    def get_top(self):
        elem = self.p_que.get().item
        
        while not self.p_que.empty():
            elem_prime = self.p_que.queue[0].item   # peek

            if elem_prime.rad == elem.rad:
                # pop out elements of same radius
                self.p_que.get()
            else:
                break

        return elem.env, elem.s, elem.a, elem.rad, elem.re, elem.no, elem.gid, elem.node_id


    def update_map(self, rad):
        self.certify_map[rad] = self.re_min
        self.log_func(f'------------------------ putting elem into certify_map: {rad} : {self.re_min}')


    def peek_top(self):
        elem = self.p_que.queue[0].item     # peek

        return elem.rad, elem.gid


    def run(self, env, state):
        self.expand(env, state, rad_lim=0, re_cur=0, no=0, node_id=0)

        while 1:
            if self.p_que.empty():
                break
            env, s, a, rad, re, no, gid, node_id = self.get_top()

            self.state_dict = dict()

            self.log_func(f'start from {no} with rad={rad} and re={re}')

            self.update_map(rad)

            if self.p_que.empty():
                break
            rad_prime, gid_prime = self.peek_top()

            assert rad_prime > rad, f'rad_prime ({rad_prime}) <= rad ({rad}), gid_prime ({gid_prime}) gid ({gid})'

            next_state, reward, done, _ = env.step(a)

            self.node_id += 1
            self.edges.append((node_id, self.node_id, a, reward, rad_prime))

            # reward shaping for Pong
            reward = max(reward, 0)

            assert not done

            self.expand(env, next_state, rad_prime, re+reward, no+1, self.node_id)

    def save(self, filename):
        torch.save(self.certify_map, filename)
        self.log_func(f'certify map saved to {filename}')

    def save_tree(self, filename):
        torch.save(self.edges, filename)
        self.log_func(f'tree edges saved to {filename}')

