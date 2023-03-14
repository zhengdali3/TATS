# import torch
import math

import cupy
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import softmax

def getGaussian(T, H, W, beta, x, y, z):
    weight = cp.ones((T, H, W))
    d = cp.diag([beta[0], beta[1], beta[1]])
    rv = multivariate_normal([x , y, z], d)
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            for k in range(weight.shape[2]):
                weight[i][j][k] = rv.pdf([i, j, k])

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # print(weight, sum(sum(sum(weight))))
    # print(weight)
    weight = weight / cp.max(weight)

    return weight

    # print(y_a, y_a.shape)

def FocusedAttention(Q, K, V):
    V = cupy.asarray(V)
    B = V.shape[0]
    NH = V.shape[1]
    T = V.shape[2]
    HS = V.shape[3]

    T = 4
    H = 16
    W = 16
    encodings_dim = 256

    V_reshaped = V.reshape(B, NH, T, H, W, encodings_dim, HS)
    Q_reshaped = Q.reshape(B, NH, T, H, W, encodings_dim, HS)
    K_reshaped = K.reshape(B, NH, T, H, W, encodings_dim, HS)
    A_reshaped = cupy.zeros((B, NH, T, H, W, encodings_dim, HS))

    for b in range(B):
        for nh in range(NH):
                for t in range(T):
                    for h in range(H):
                        for w in range(W):
                            for enc in range(encodings_dim):
                                # q shape is (HS,)
                                q = Q_reshaped[b][nh][t][h][w][enc][:]
                                
                                # k and v shape is (T, H, W, HS)
                                k = K_reshaped[b][nh][:][:][:][enc][:]
                                v = V_reshaped[b][nh][:][:][:][enc][:]
                                
                                # boardcast for multiply
                                q_reshaped = q.reshape((1, 1, 1, HS))
                                qk = k * q_reshaped
                                qk = qk.reshape(-1, HS)
                                # qk shape should be (T*H*W, HS)
                                qk = cupy.sum(qk, axis=1)
                                

                                # score shape is (T*H*W)
                                score = softmax(qk / math.sqrt(HS))
                                score = score[:, np.newaxis]

                                weight = getGaussian(T, H, W, [100, 100], t, h, w)
                                weight = weight[:, np.newaxis]
                                v = v * weight

                                # v shape is (T*H*W, HS)
                                v = v.reshape(-1, HS)
                                v = v * score
                                a = cupy.sum(v, axis=0)
                                A_reshaped[b][nh][t][h][w][enc][:] = a
    A = A_reshaped.reshape(B,NH,T,HS)
    return A

getGaussian(4,4,4,[100,100],1,1,1)

import cupy as cp

# create an example input tensor with shape (B, nh, hs, encodings_dim, t, h, w)
input_tensor = cp.random.randn(2, 3, 4, 5, 6, 7, 8)

# create an example matrix to multiply with, with shape (t, h, w)
mul_matrix = cp.random.randn(5, 6, 7)

# Reshape the input tensor and multiplication matrix to enable broadcasting
input_tensor_reshaped = cp.reshape(input_tensor, (input_tensor.shape[0]*input_tensor.shape[1]*input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4], input_tensor.shape[5], input_tensor.shape[6]))
mul_matrix_reshaped = cp.broadcast_to(cp.expand_dims(cp.expand_dims(cp.expand_dims(mul_matrix, axis=0), axis=0), axis=0), (input_tensor[0]*input_tensor.shape[1]*input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4], input_tensor.shape[5], input_tensor.shape[6]))

# perform the element-wise multiplication of the submatrices in one step
output_tensor = cp.multiply(input_tensor_reshaped, mul_matrix_reshaped)

# print the first element of the modified input tensor for validation
print("First element of the modified input tensor:")
print(output_tensor[0, 0, 0, :, :, :, :])
