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

    T, H, W = 4, 4, 4
    T_flatten = T*H*W
    center_T, center_H, center_W = T-1, H-1, W-1
    beta = [100, 100]

    # getGaussian(T, H, W, beta, )

    @staticmethod
    def forward(ctx, score, V):

        att=[]

        for pos in numpy.arange(0, focusAttention.T_flatten):
            print(f"start of loop {torch.cuda.memory_allocated()}")

            i = math.floor(pos / (focusAttention.H * focusAttention.W))
            j = math.floor((pos - i * focusAttention.H * focusAttention.W) / focusAttention.H)
            k = pos - i * focusAttention.H * focusAttention.W - j * focusAttention.W

            print(f"Before weight_xyz {torch.cuda.memory_allocated()}")

            weight_xyz = weight[focusAttention.center_T - i:2 * focusAttention.center_T - i + 1, focusAttention.center_W - j:2 * focusAttention.center_W - j + 1,
                         focusAttention.center_H - k:2 * focusAttention.center_H - k + 1].reshape(-1)

            print(f"After sub indexing weight {torch.cuda.memory_allocated()}")

            weight_xyz = weight_xyz[None, None, :, None]

            print(f"After add axis {torch.cuda.memory_allocated()}")

            # V_focused = V * weight_xyz

            print(f"After multiply weight {torch.cuda.memory_allocated()}")

            # qk shape (B, NH, 1, T)
            qk = score[:, :, pos, :]

            print(f"After index qk{torch.cuda.memory_allocated()}")

            qk = qk[:, :, None, :]

            print(f"After add axis to qk {torch.cuda.memory_allocated()}")

            att_pos = qk @ (V * weight_xyz)

            print(f"After qk @ V {torch.cuda.memory_allocated()}")

            att.append(att_pos)

            print(f"After att append {torch.cuda.memory_allocated()}")

            # V = torch.clone(V_ori)

        print(f"Before cat {torch.cuda.memory_allocated()}")

        result = torch.cat(att, dim=2)
        end = torch.cuda.memory_allocated()
        print(f"After focused attention, memory usage is {end}, memory used {end - st}")

        ctx.save_for_backward(score, V, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        score, V, result = ctx.saved_tensor

        grad_score = []
        grad_V = []

        for pos in numpy.arange(0, focusAttention.T_flatten):
            grad_att_pos = grad_output[:, pos, :, :]

            i = math.floor(pos / (focusAttention.H * focusAttention.W))
            j = math.floor((pos - i * focusAttention.H * focusAttention.W) / focusAttention.H)
            k = pos - i * focusAttention.H * focusAttention.W - j * focusAttention.W

            weight_xyz = weight[focusAttention.center_T - i:2 * focusAttention.center_T - i + 1, focusAttention.center_W - j:2 * focusAttention.center_W - j + 1,
                         focusAttention.center_H - k:2 * focusAttention.center_H - k + 1].reshape(-1)

            V_focus = V * weight_xyz

            qk = score[:, :, pos, :]

            grad_qk = grad_att_pos @ torch.linalg.inv(V_focus)
            grad_V_focus = torch.linalg.inv(qk) @ grad_att_pos

            grad_score.append(grad_qk)
            grad_V_focus = grad_V_focus * weight_xyz
            grad_V.append(grad_V_focus)

        grad_V_focus



    