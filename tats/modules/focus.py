import gc

import torch
import math
import cupy
import numpy as np
import cupy as cp
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

    T, H, W = 4, 4, 4
    T_flatten = T * H * W
    center_T, center_H, center_W = T - 1, H - 1, W - 1
    beta = [100, 100]
    device = torch.device("cuda:0")
    weight = getGaussian(T, H, W, beta, device)

    @staticmethod
    def forward(ctx, score, V):

        att=[]

        st = torch.cuda.memory_allocated()

        for pos in np.arange(0, focusAttention.T_flatten):

            # print(f"start of loop {torch.cuda.memory_allocated()}")

            i = math.floor(pos / (focusAttention.H * focusAttention.W))
            j = math.floor((pos - i * focusAttention.H * focusAttention.W) / focusAttention.H)
            k = pos - i * focusAttention.H * focusAttention.W - j * focusAttention.W

            # print(f"Before weight_xyz {torch.cuda.memory_allocated()}")

            weight_xyz = focusAttention.weight[focusAttention.center_T - i:2 * focusAttention.center_T - i + 1, focusAttention.center_W - j:2 * focusAttention.center_W - j + 1,
                         focusAttention.center_H - k:2 * focusAttention.center_H - k + 1].reshape(-1)

            # print(f"After sub indexing weight {torch.cuda.memory_allocated()}")

            weight_xyz = weight_xyz[None, None, :, None]

            # print(f"After add axis {torch.cuda.memory_allocated()}")

            # V_focused = V * weight_xyz

            # print(f"After multiply weight {torch.cuda.memory_allocated()}")

            # qk shape (B, NH, 1, T)
            qk = score[:, :, pos, :]

            # print(f"After index qk{torch.cuda.memory_allocated()}")

            qk = qk[:, :, None, :]

            # print(f"After add axis to qk {torch.cuda.memory_allocated()}")

            att_pos = torch.matmul(qk, (V * weight_xyz)).detach()

            # print(f"After qk @ V {torch.cuda.memory_allocated()}")

            att.append(att_pos)

            # print(f"After att append {torch.cuda.memory_allocated()}")

            gc.collect()
            # V = torch.clone(V_ori)


        # print(f"Before cat {torch.cuda.memory_allocated()}")

        result = torch.cat(att, dim=2).detach()

        end = torch.cuda.memory_allocated()

        # print(f"result memory usage is {result.element_size() * result.nelement()}, memory used {end - st}, memory for v is {V.element_size() * V.nelement()}")

        # print(f"After focused attention, memory usage is {end}, memory used {end - st}")

        torch.cuda.empty_cache()

        # print(f"After empty cache, memory usage is {torch.cuda.memory_allocated()}")

        ctx.save_for_backward(score, V, result)

        # print(f"After save for backwards, memory usage is {torch.cuda.memory_allocated()}")

        return result

    @staticmethod
    def backward(ctx, grad_output):
        score, V, result = ctx.saved_tensor

        grad_score = []
        grad_V = []

        for pos in np.arange(0, focusAttention.T_flatten):
            grad_att_pos = grad_output[:, :, pos, :]

            grad_att_pos = grad_att_pos[:, :, None, :]

            i = math.floor(pos / (focusAttention.H * focusAttention.W))
            j = math.floor((pos - i * focusAttention.H * focusAttention.W) / focusAttention.H)
            k = pos - i * focusAttention.H * focusAttention.W - j * focusAttention.W

            weight_xyz = focusAttention.weight[focusAttention.center_T - i:2 * focusAttention.center_T - i + 1, focusAttention.center_W - j:2 * focusAttention.center_W - j + 1,
                         focusAttention.center_H - k:2 * focusAttention.center_H - k + 1].reshape(-1)

            qk = score[:, :, pos, :]

            qk = torch.swapaxes(qk, 2, 3)

            grad_V.append(torch.matmul((qk * weight_xyz), grad_att_pos)[:, :, pos, :])

            grad_score.append(torch.matmul(weight_xyz, grad_att_pos) * V[:, :, :, 0])

            # grad_qk = grad_att_pos @ torch.linalg.inv(V_focus)
            # grad_V_focus = torch.linalg.inv(qk) @ grad_att_pos
            #
            # grad_score.append(grad_qk)
            # grad_V_focus = grad_V_focus * weight_xyz
            # grad_V.append(grad_V_focus)

        # Shape should be B, NH, T, T
        grad_score = torch.cat(grad_score, dim=2)

        # Shape should be B, NH, T, HS
        grad_V = torch.cat(grad_V, dim=2)

        return grad_score, grad_V

focus = focusAttention.apply

from torch.autograd import gradcheck

input = (torch.randn(4, 4, 64, 64, dtype=torch.float64, requires_grad=True, device='cuda:0'), torch.randn(4, 4, 64, 4, dtype=torch.float64, requires_grad=True, device='cuda:0'))
# print("memory usage before apply is {}".format(torch.cuda.memory_allocated()))
# st = torch.cuda.memory_allocated()
# focus = focusAttention.apply(input[0], input[1])
# print("memory usage is after focus is {}".format(torch.cuda.memory_allocated() - st))

test = gradcheck(focus, input, eps=1e-6, atol=1e-4)
# print(test)


    