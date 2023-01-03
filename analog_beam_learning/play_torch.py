import torch
import numpy as np


def phase2bf(ph_mat):
    # ph_mat: (i) a tensor, (ii) B x N
    # bf_mat: (i) a tensor, (ii) B x 2N

    # N = ph_mat.shape[1]
    bf_mat = torch.exp(1j * ph_mat)
    bf_mat_r = torch.real(bf_mat)
    bf_mat_i = torch.imag(bf_mat)

    bf_mat_ = torch.cat((bf_mat_r, bf_mat_i), dim=1)

    return bf_mat_


x = torch.rand(1, 10) * 2 * np.pi
x = x.float().cuda()

y = torch.exp(1j * x)
y_r = torch.real(y)
y_i = torch.imag(y)

z = torch.randn(10, 1) + 1j * torch.randn(10, 1)
z = z.cuda()

w = y @ z

xx = torch.randn(10, 1)
yy = torch.min(xx, dim=1).values.reshape(-1, 1)

xxx = torch.randn(3, 4)

pp = 1
