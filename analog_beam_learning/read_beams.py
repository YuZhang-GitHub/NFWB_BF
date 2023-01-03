import numpy as np
import scipy.io as scio

num_ant = 256
num_beam = 1
results = np.empty((num_beam, num_ant))

path = './beams/'

for beam_id in range(num_beam):
    fname = 'beams_' + str(beam_id) + '_max_trsize_2000_epoch_500_3bit.txt'
    with open(path + fname, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        results[beam_id, :] = np.fromstring(last_line.replace("\n", ""), sep=',').reshape(1, -1)

scio.savemat('ULA_PS_only.mat', {'beams': results})
