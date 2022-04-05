import torch
from tqdm import tqdm
import time


class EstRange:
    def __init__(self, log_func, input_shape, model, forward_func, m=10000, sigma=0.01):
        self.log_func = log_func
        self.model = model
        self.forward_func = forward_func
        self.m = m
        self.sigma = sigma

        self.log_func(f'start generating noise list!')
        t = time.time()
        self.noise_list = [ torch.FloatTensor(*input_shape).normal_(0, self.sigma).cuda()
                        for _ in tqdm(range(self.m)) ]
        print(self.noise_list[0].shape)

        self.log_func(f'generating {self.m} noise done using {time.time() - t} seconds!')

        self.init_record_list()


    def init_record_list(self):
        self.q_range = []


    def forward(self, state, epsilon=0, perturb=-1, return_q=False):

        with torch.no_grad():
            q_value = torch.mean(torch.stack(
                [ self.forward_func(state + self.noise_list[i]) for i in tqdm(range(self.m)) ]), dim=0)

        action  = q_value.max(1)[1].data.cpu().numpy()
#         print(action)

        q_max = torch.max(q_value).item()
        q_min = torch.min(q_value).item()
#         print(q_max, q_min)

        self.q_range.append((q_max, q_min))

        return action, q_value if return_q else action


    def save(self, filename):
        torch.save(self.q_range, filename)
        self.log_func(f'q range saved to {filename}')

        # reset
        self.init_record_list()


