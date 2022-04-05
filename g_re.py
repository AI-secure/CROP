import torch
import torch.nn as nn


class GRe(nn.Module):
    def __init__(self, log_func, input_shape, model, forward_func, sigma=0.01):
        super(GRe, self).__init__()
        self.log_func = log_func
        self.input_shape = input_shape
        self.model = model
        self.forward_func = forward_func
        self.sigma = sigma


    @torch.no_grad()
    def forward(self, state, return_q=False):
        noise = torch.FloatTensor(*(self.input_shape)).normal_(0, self.sigma).cuda()

        q_value = self.forward_func(state + noise)
        action  = q_value.max(1)[1].data.cpu().numpy()

        return action, q_value if return_q else action

    def forward_requires_grad(self, state, return_q=False):
        noise = torch.FloatTensor(*(self.input_shape)).normal_(0, self.sigma).cuda()

        q_value = self.forward_func(state + noise)
        action  = q_value.max(1)[1].data.cpu().numpy()

        return action, q_value if return_q else action
