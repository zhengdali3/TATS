import torch
import math
import cupy
import numpy as np
import cupy as cp
from datetime import datetime
from scipy.stats import multivariate_normal
from scipy.special import softmax
from torch.autograd import Function


def getGaussian(T, H, W, beta, d):

    diag = np.diag([beta[0], beta[1], beta[1]])
    rv = multivariate_normal([T - 1, H - 1, W - 1], diag)
    tensor = torch.tensor((), dtype=torch.float32)

    NT = 2 * T - 1
    NH = 2 * H - 1
    NW = 2 * W - 1

    weight = tensor.new_ones((NT, NW, NH), device=d)

    for pos in np.arange(0, NT * NH * NW):
        i = math.floor(pos / (NH * NW))
        j = math.floor((pos - i * NH * NW) / NH)
        k = pos - i * NH * NW - j * NW
        weight[i, j, k] = rv.pdf([i, j, k])

        weight = weight / torch.max(weight)

    return weight

class focusAttention(Function):

    T, H, W = 4, 16, 16
    T_flatten = T * H * W
    center_T, center_H, center_W = T - 1, H - 1, W - 1
    beta = [10000, 10000]
    
    diag = np.diag([beta[0], beta[1], beta[1]])
    rv = multivariate_normal([T - 1, H - 1, W - 1], diag)
    tensor = torch.tensor((), dtype=torch.float32)

    NT = 2 * T - 1
    NH = 2 * H - 1
    NW = 2 * W - 1

    weight_cuda0 = tensor.new_ones((NT, NW, NH), device=torch.device("cuda:0"))

    for pos in np.arange(0, NT * NH * NW):
        i = math.floor(pos / (NH * NW))
        j = math.floor((pos - i * NH * NW) / NH)
        k = pos - i * NH * NW - j * NW
        weight_cuda0[i, j, k] = rv.pdf([i, j, k])

    weight_cuda0 = weight_cuda0 / torch.max(weight_cuda0)
    
    # print(weight_cuda0, weight_cuda0[T-1, H-1, W-1])
    
    # Shape T, 1, 1, T, 1
    V_weight_cuda0 = torch.empty((T_flatten,1,1,T_flatten,1), dtype=torch.float32, device ="cuda:0")

    for pos in np.arange(0, T_flatten):

        i = math.floor(pos / (H * W))
        j = math.floor((pos - i * H * W) / H)
        k = pos - i * H * W - j * W

        weight_xyz = weight_cuda0[center_T - i:2 * center_T - i + 1, center_W - j:2 * center_W - j + 1,
                     center_H - k:2 * center_H - k + 1].reshape(-1)

        V_weight_cuda0[pos, 0, 0, :, 0] = weight_xyz

    V_weight_cuda1 = V_weight_cuda0.detach().to("cuda:1")

    @staticmethod
    def forward(ctx, score, V):
        
        B, NH, T_flatten, HS = V.shape
        d = V.get_device()
        
        if d == 0:
            V_weight = focusAttention.V_weight_cuda0
        else:
            V_weight = focusAttention.V_weight_cuda1
        
        V_full = V_weight * V

        qk = score[:, :, :, None, :]
        
        # qk should be T, B, NH, 1 , T
        qk = qk.permute(2, 0, 1, 3, 4)

        # result should be T, B, NH, 1, HS
        result = torch.empty((T_flatten, B, NH, 1, HS), dtype = torch.float32, device = f"cuda:{d}")
        
        div = 16
        for sub in np.arange(div):
            base = int(focusAttention.T_flatten / div)
            result[base*sub:base*(sub+1), :, :, :, :] = torch.matmul(qk[base*sub:base*(sub+1), :, :, :, :], V_full[base*sub:base*(sub+1), :, :, :, :])
        
        result = result[:, :, :, 0, :].permute(1, 2, 0, 3)
        
        ctx.save_for_backward(score, V, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        
        st_loop = datetime.now()
        
        score, V, result = ctx.saved_tensors
        
        B, NH, T_flatten, HS = V.shape

        # V weight shape is [1, 1, T, T]
        d = V.get_device()

        if d == 0:
            V_weight = focusAttention.V_weight_cuda0
        else:
            V_weight = focusAttention.V_weight_cuda1

        # （1, B, NH, T, T)
        score = score[None, :, :, :, :]

        # (T, B, NH, T, 1)
        score = score.permute(3, 1, 2, 4, 0)

        # (1, B, NH, T, HS)
        grad_output_V = grad_output[None, :, :, :, :]

        # (T, B, NH, 1, HS)
        grad_output_V = grad_output_V.permute(3, 1, 2, 0, 4)

        grad_V = torch.empty(V.shape, dtype = torch.float32, device = f"cuda:{d}")
        grad_V_total = torch.empty((T_flatten, B, NH, T_flatten, HS), dtype = torch.float32, device = f"cuda:{d}")
        
        div = 16
        base = int(focusAttention.T_flatten / div)
        for sub in np.arange(div):
            grad_V_total[base*sub:base*(sub+1), :, :, :, :] = torch.matmul((V_weight * score)[base*sub:base*(sub+1), :, :, :, :], grad_output_V[base*sub:base*(sub+1), :, :, :, :])
        del grad_output_V
        
        for i in np.arange(focusAttention.T_flatten):
            grad_V[:, :, i, :] = grad_V_total[i, :, :, i, :]
        del grad_V_total

        grad_score = V[:, :, :, 0][:, :, :, None] * grad_output[:, :, :, 0][:, :, :, None] * V_weight

        # (T, B, NH, T)
        grad_score = grad_score[:, :, :, :, 0]

        # (B, NH, T, T)
        grad_score = grad_score.permute(1, 2, 0, 3)
        
        end = datetime.now()
        
        return grad_score, grad_V
