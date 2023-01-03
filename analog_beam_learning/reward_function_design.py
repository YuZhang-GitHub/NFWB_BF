import torch
import numpy as np
import matplotlib.pyplot as plt


def reward_func_(x):
    y = torch.zeros(x.size()).float().cuda()
    ind = torch.abs(x) > 1e-10
    y[ind] = torch.div(torch.tensor(1.), 1000 * x[ind])
    y[~ind] = x[~ind]
    return y


def reward_func(x, thres=0.1):
    y = torch.zeros(x.size()).float().cuda()
    ind = torch.abs(x) < thres
    y[ind] = x[ind] * 100
    y[~ind] = x[~ind]
    return y


# def reward_func_v2(x):
#     pi_over_2 = torch.tensor(np.pi/2).float().cuda()
#     y = torch.zeros(x.size()).float().cuda()
#     ind = torch.abs(x) < 0.5
#     y[ind] = x[ind] * 100
#     y[~ind] = torch.tan(pi_over_2*x[~ind])
#     return y


def reward_func_v2(x):
    return x * 100


if __name__ == '__main__':
    x = np.linspace(0.00001, 1.00001, num=100000)

    y = np.log10(1 / (1 - x))

    plt.plot(x, y)

    plt.show()
