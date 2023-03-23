import torch
import math
import cupy
import numpy as np
import cupy as cp
from scipy.stats import multivariate_normal
from scipy.special import softmax
from torch.autograd import Function

class focusAttention(Function):
    
    def getGaussian(T, H, W, beta, d):
        diag = numpy.diag([beta[0], beta[1], beta[1]])
        rv = multivariate_normal([T-1, H-1, W-1], diag)
        tensor = torch.tensor((), dtype=torch.float32)

        NT = 2*T-1
        NH = 2*H-1
        NW = 2*W-1

        weight = tensor.new_ones((NT, NW, NH), device=d)

        for pos in numpy.arange(0, NT*NH*NW):
            i = math.floor(pos/(NH*NW))
            j = math.floor((pos - i * NH * NW) / NH)
            k = pos - i * NH * NW - j * NW
            # print(f"i {i}, j {j}, k {k}")
            weight[i, j, k] = rv.pdf([i, j, k])

            weight = weight / torch.max(weight)

        return weight
    
    @staticmethod
    def forward(ctx, input):
    