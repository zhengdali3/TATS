# import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def getGaussian(t, h, w, beta, x, y, z):
    weight = np.ones((t, h, w))
    d = np.diag([beta[0], beta[1], beta[1]])
    rv = multivariate_normal([x , y, z], d)
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            for k in range(weight.shape[2]):
                weight[i][j][k] = rv.pdf([i, j, k])

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    print(weight, sum(sum(sum(weight))))

    weight = weight / np.max(weight)

    print(weight)
    # print(y_a, y_a.shape)

getGaussian(4,4,4,[100,100],1,1,1)
