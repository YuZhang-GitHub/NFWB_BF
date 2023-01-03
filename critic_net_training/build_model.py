import torch
import torch.nn as nn


class LearningModel(nn.Module):

    def __init__(self, in_size, ou_size):
        super(LearningModel, self).__init__()

        self.M = int(in_size / 2)
        self.output_size = ou_size

        # just for debugging
        # self.ch_r = torch.from_numpy(ch[:, :self.M].transpose()).float()
        # self.ch_i = torch.from_numpy(ch[:, self.M:].transpose()).float()

        self.H_r = nn.Parameter(torch.randn(self.M, 2), requires_grad=True)
        self.H_i = nn.Parameter(torch.randn(self.M, 2), requires_grad=True)

    def forward(self, state_action_pair):

        # x = torch.cat((state, action), 1)

        state = state_action_pair[:, :self.M]
        action = state_action_pair[:, self.M:]

        x_state = phase2bf(state)
        x_state_r, x_state_i = x_state[:, :self.M], x_state[:, self.M:]

        z_r = x_state_r @ self.H_r + x_state_i @ self.H_i
        z_i = x_state_i @ self.H_r - x_state_r @ self.H_i

        z = z_r ** 2 + z_i ** 2
        z_min = torch.mean(z, dim=1).reshape(-1, 1)

        x_action = phase2bf(action)
        x_action_r, x_action_i = x_action[:, :self.M], x_action[:, self.M:]

        u_r = x_action_r @ self.H_r + x_action_i @ self.H_i
        u_i = x_action_i @ self.H_r - x_action_r @ self.H_i

        u = u_r ** 2 + u_i ** 2
        u_min = torch.mean(u, dim=1).reshape(-1, 1)

        out = 10 * torch.log10(u_min) - 10 * torch.log10(z_min)

        return out


def phase2bf(ph_mat):
    # ph_mat: (i) a tensor, (ii) B x M
    # bf_mat: (i) a tensor, (ii) B x 2M
    # B stands for batch size and M is the number of antenna

    M = torch.tensor(ph_mat.shape[1]).to(ph_mat.device)
    bf_mat = torch.exp(1j * ph_mat)
    bf_mat_r = torch.real(bf_mat)
    bf_mat_i = torch.imag(bf_mat)

    bf_mat_ = (1 / torch.sqrt(M)) * torch.cat((bf_mat_r, bf_mat_i), dim=1)

    return bf_mat_
