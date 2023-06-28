import logging
import argparse

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim

from paddle.distribution import Beta
from paddle.distribution import Uniform


import numpy as np

from copy import deepcopy




def train_distribution_diversity(n_distribution, n_dim, max_iters=1000):

    n_mean = nn.Parameter(paddle.randn(n_distribution, n_dim))
    temp_matrix = 1 - paddle.eye(int(n_dim), dtype=paddle.float32, requires_grad=False)
    optimizer = optim.SGD([n_mean], lr=0.5, momentum=0.9)
    for i in range(max_iters):
        normed_x = n_mean / n_mean.norm(dim=1).unsqueeze(1)
        cov = paddle.mm(normed_x.t(), normed_x)**2 / (n_distribution - 1)
        # cov = paddle.mm(normed_x.t(), normed_x)**2 / (n_distribution - n_dim)
        loss = paddle.mean(cov * temp_matrix)
        loss.backward()
        optimizer.step()
        print(f"Optimizing diverse distribution... n_distribution:{n_distribution}, n_dim:{n_dim}\
                    Iter: {i}, loss: {loss.item()}")
        # logging.debug(f"Optimizing diverse distribution... n_distribution:{n_distribution}, n_dim:{n_dim}\
        #             Iter: {i}, loss: {loss.item()}")

    normed_n_mean = n_mean / n_mean.norm(dim=1).unsqueeze(1)
    return normed_n_mean





"""
Modified from:
https://github.com/moukamisama/F2M/tree/main
"""

class DiscreteUniform():
    def __init__(self, device, bound, shape, reduction_factor):
        self.device = device
        self.bound = bound
        self.shape = shape
        b = int(reduction_factor / 2)
        ratio = paddle.arange(-b, b+1, device=self.device)
        self.choice = ratio / reduction_factor
        self.choice = self.choice * bound
        # self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')


    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # for r in self.choice:
        #     sum = (noise == r).sum()
        #     self.logger.info(f'{r}: {sum/len(noise)}')
        return noise

    def generate_pos(self):
        return paddle.randint(len(self.choice), self.shape, device=self.device)

class DiscreteUniform2():
    def __init__(self, device, bound, shape, reduction_factor):
        self.device = device
        self.bound = bound
        self.shape = shape
        ratio1 = paddle.to_tensor([-2, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2], place=self.device)
        ratio2 = paddle.to_tensor([-2, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2], place=self.device)
        self.choice = ratio1 / 4
        self.choice = self.choice * bound
        # self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')

    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # str = ''
        # unique = paddle.unique_consecutive(self.choice)
        # for r in unique:
        #     sum = (noise == r).sum()
        #     str += f'!!! {sum/len(noise):.5f} '
        # self.logger.info(str)
        return noise

    def generate_pos(self):
        return paddle.randint(len(self.choice), self.shape, device=self.device)

class DiscreteUniform3():
    def __init__(self, device, bound, shape, reduction_factor):
        self.device = device
        self.bound = bound
        self.shape = shape
        ratio1 = paddle.to_tensor([-2, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2], place=self.device)

        self.choice = ratio1 / 4
        self.choice = self.choice * bound
        # self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')

    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # str = ''
        # unique = paddle.unique_consecutive(self.choice)
        # for r in unique:
        #     sum = (noise == r).sum()
        #     str += f'!!! {sum/len(noise):.5f} '
        # self.logger.info(str)
        return noise

    def generate_pos(self):
        return paddle.randint(len(self.choice), self.shape, device=self.device)

class BetaDistribution():
    def __init__(self, device, alpha, beta, upper_bound, lower_bound):
        self.device = device
        self.m = Beta(alpha, beta)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.length = self.upper_bound - self.lower_bound

    def sample(self):
        ans = self.m.sample()
        ans = ans * self.length + self.lower_bound
        return ans

class DiscreteBetaDistribution():
    def __init__(self, device, low, high, shape, bound, reduction_factor):
        self.device = device
        self.low = low
        self.high = high
        self.shape = shape

        self.bound = bound
        b = int(reduction_factor / 2)
        ratio = paddle.arange(-b, b + 1, device=self.device)
        self.choice = ratio / reduction_factor
        self.anchors = paddle.arange(self.choice.shape[0]+1) / (self.choice.shape[0])
        self.choice = self.choice * bound

        self.alpha = paddle.to_tensor([0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], place=self.device)
        self.beta = paddle.to_tensor([0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], place=self.device)

    def sample(self):
        a_p = np.random.randint(7)
        b_p = np.random.randint(7)
        alpha = self.alpha[a_p]
        beta = self.beta[b_p]
        alpha = paddle.full(self.shape, alpha, device=self.device)
        beta = paddle.full(self.shape, beta, device=self.device)

        m = Beta(alpha, beta)

        ans = m.sample()

        for i in range(self.anchors.shape[0]-1):
            a1 = self.anchors[i]
            a2 = self.anchors[i+1]
            p = paddle.logical_and(ans >= a1, ans <= a2)
            ans[p] = i
        ans = ans.long()
        ans = self.choice[ans]

        return ans

class BoundNormal():
    def __init__(self, device, mean, sigma, bound_value):
        self.device = device
        self.m = paddle.distribution.Normal(mean, sigma)
        self.mean = mean
        self.sigma = sigma
        self.bound_value = bound_value

    def sample(self):
        ans = self.m.sample()
        upper_bound = self.bound_value / 2
        lower_bound = - self.bound_value / 2
        upper_mask = paddle.greater_than(ans, upper_bound)
        lower_mask = paddle.less_than(ans, lower_bound)

        # n_upper_pos = upper_mask.sum()
        # n_lower_pos = lower_mask.sum()
        # n_total_pos = paddle.ones(ans.shape).sum()
        #
        # if n_lower_pos > 0 or n_lower_pos < 0:
        #     logger = get_root_logger()
        #     logger.debug(f'mean:{self.mean}, sigma:{self.sigma}, n_upper_pos: {n_upper_pos}, n_lower_pos: {n_lower_pos}, total_pos: {n_total_pos}')

        ans[upper_mask] = upper_bound[upper_mask]
        ans[lower_mask] = lower_bound[lower_mask]

        return ans

class BoundUniform():
    def __init__(self, device, low, high, bound_value):
        self.device = device
        self.m = Uniform(low, high)
        self.low = low
        self.high = high
        self.bound_value = bound_value

    def sample(self):
        ans = self.m.sample()
        upper_bound = self.bound_value / 2
        lower_bound = - self.bound_value / 2
        upper_mask = paddle.greater_than(ans, upper_bound)
        lower_mask = paddle.less_than(ans, lower_bound)

        # n_upper_pos = upper_mask.sum()
        # n_lower_pos = lower_mask.sum()
        # n_total_pos = paddle.ones(ans.shape).sum()
        #
        # if n_lower_pos > 0 or n_lower_pos < 0:
        #     logger = get_root_logger()
        #     logger.debug(f'low:{self.low}, high:{self.high}, n_upper_pos: {n_upper_pos}, n_lower_pos: {n_lower_pos}, total_pos: {n_total_pos}')

        ans[upper_mask] = upper_bound[upper_mask]
        ans[lower_mask] = lower_bound[lower_mask]

        return ans

class UniformElements():
    def __init__(self, device, low, high, params_shape):
        self.device = device
        self.m = Uniform(low, high)
        self.low = low
        self.high = high
        self.params_shape = params_shape

    def sample(self):
        # TODO finish element-wise Uniform distribution
        return

def pnorm(data, p):
    normB = paddle.norm(data, 2, dim=1)
    for i in range(data.shape[0]):
        data[i] = data[i] / paddle.pow(normB[i], p)
    return data




