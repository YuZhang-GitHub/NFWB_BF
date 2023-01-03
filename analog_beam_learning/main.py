import os
import random
import torch
import numpy as np
import scipy.io as scio

from train_ddpg import train
from util_lib import load_ch, bf_gain_calc


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if __name__ == '__main__':

    options = {
        'gpu_idx': 0,
        'num_ant': 256,
        'num_beams': 1,
        'num_bits': 3,
        'num_NNs': 1,
        'num_loop': 1,  # outer loop
        'target_update': 3,
        'pf_print': 10,

        'beam_idx': 0,

        'path': './H_fc.mat',

        'save_freq': 50000
    }

    train_opt = {
        'state': 0,
        'best_state': 0,
        'num_iter': 100,
        'tau': 1e-2,
        'overall_iter': 1,
        'replay_memory': [],
        'replay_memory_size': 256,
        'minibatch_size': 256,
        'gamma': 0
    }

    if not os.path.exists('beams/'):
        os.mkdir('beams/')

    if not os.path.exists('pfs/'):
        os.mkdir('pfs/')

    ch = load_ch(options['path'])  # numpy.ndarray: (# of user, 2 * # of ant)

    options['H_r'] = scio.loadmat('../critic_net_training/critic_params_trsize_2000_epoch_500_3bit.mat')['H_r']
    options['H_i'] = scio.loadmat('../critic_net_training/critic_params_trsize_2000_epoch_500_3bit.mat')['H_i']

    options['exp_name'] = 'trsize_2000_epoch_500_3bit'

    # -----------------------------------------------------------------------------

    X = ch

    target = np.array([1.])  # statistics interested
    options['target'] = target

    # ------------------------------- Quantization settings ---------------------------------------------- #

    options['num_ph'] = 2 ** options['num_bits']
    options['multi_step'] = torch.from_numpy(
        np.linspace(int(-(options['num_ph'] - 2) / 2),
                    int(options['num_ph'] / 2),
                    num=options['num_ph'],
                    endpoint=True)).float().reshape(1, -1)
    options['pi'] = torch.tensor(np.pi)
    options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
    options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)

    for beam_id in range(options['num_NNs']):
        train(X, options, train_opt, beam_id)
