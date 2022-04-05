import numpy as np
from scipy.stats import norm
import time
import torch
import torch.nn as nn
from tqdm import tqdm


class LoAct(nn.Module):
    def __init__(self, log_func, input_shape, model, forward_func, m=10000, sigma=0.01, v_lo=0, v_hi=1, conf_bound=0.05):
        super(LoAct, self).__init__()
        self.log_func = log_func
        self.model = model
        self.forward_func = forward_func
        self.m = m
        self.sigma = sigma
        self.v_lo = v_lo
        self.v_hi = v_hi
        self.delta = ((self.v_hi - self.v_lo) * np.sqrt(np.log(1 / conf_bound) / (2 * self.m)))
        self.log_func(f'v_lo = {v_lo}, v_hi = {v_hi}, delta = {self.delta}')

        self.log_func(f'start generating noise list!')
        t = time.time()
        self.noise_list = [ torch.FloatTensor(*input_shape).normal_(0, self.sigma).cuda()
                        for _ in tqdm(range(self.m)) ]
        self.log_func(f'generating {self.m} noise done using {time.time() - t} seconds!')

        self.init_record_list()


    def init_record_list(self):
        self.R_list = []
        self.tot_steps = 0
        self.tot_cert = 0        


    @torch.no_grad()
    def forward(self, state, epsilon=0, return_q=False, cert=True):
        self.tot_steps += 1

        # step 1. rand smoothing
        # print('----- step 1 -----')

        q_value = torch.mean(torch.stack(
            [ torch.clamp(self.forward_func(state + self.noise_list[i]), min=self.v_lo, max=self.v_hi) \
                for i in tqdm(range(self.m))]), dim=0)
        q_value = torch.clamp(q_value, min=self.v_lo, max=self.v_hi)    # in case of floating point error

        top_1, top_2 = q_value.topk(2)[0].squeeze().cpu()


        # step 2. certification
        # print('----- step 2 -----')

        if cert:
            self.log_func(f'top_1 value = {top_1}, top_2 value = {top_2}')
            if top_1 - self.delta >= top_2 + self.delta:
                R = self.sigma / 2 * (
                                    norm.ppf((top_1 - self.delta - self.v_lo) / (self.v_hi - self.v_lo)) - 
                                    norm.ppf((top_2 + self.delta - self.v_lo) / (self.v_hi - self.v_lo))
                                    )
                self.tot_cert += 1
                self.R_list.append((top_1, top_2, R))
                self.log_func(f'Yes, R = {R : .4f}, cert_ratio = {self.tot_cert} / {self.tot_steps}')

            else:
                self.R_list.append((top_1, top_2, -1))
                self.log_func(f'No, cert_ratio = {self.tot_cert} / {self.tot_steps}')

        action  = q_value.max(1)[1].data.cpu().numpy()

        return action, q_value if return_q else action


    def forward_requires_grad(self, state, return_q=False):
        self.tot_steps += 1

        # rand smoothing

        q_value = torch.mean(torch.stack(
            [ torch.clamp(self.forward_func(state + self.noise_list[i]), min=self.v_lo, max=self.v_hi) \
                for i in tqdm(range(self.m))]), dim=0)
        q_value = torch.clamp(q_value, min=self.v_lo, max=self.v_hi)    # in case of floating point error

        action  = q_value.max(1)[1].data.cpu().numpy()

        return action, q_value if return_q else action


    def save(self, filename):
        torch.save(self.R_list, filename)
        self.log_func(f'certified result saved to {filename}')

        # reset
        self.init_record_list()


