import random
import torch
import torch.optim as optimizer
import torch.nn as nn
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from build_model import LearningModel

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------------------- #
# ---------------------------------- Data preparation ------------------------------------ #
# ---------------------------------------------------------------------------------------- #

path = './exp_replay_buffer_3_bit.mat'
data = scio.loadmat(path)
x, y = data['state_action_pair'], data['q_value']

# ---------------------------------------------------------------------------------------- #

num_of_sample = x.shape[0]
shuffle_ind = np.random.permutation(num_of_sample)

x_shuffled = x[shuffle_ind]
y_shuffled = y[shuffle_ind]

num_tr = 2000

dataset = {'x_trn': torch.from_numpy(x_shuffled[:num_tr, :]).float(),
           'y_trn': torch.from_numpy(y_shuffled[:num_tr, :]).float(),
           'x_val': torch.from_numpy(x_shuffled[num_tr:, :]).float(),
           'y_val': torch.from_numpy(y_shuffled[num_tr:, :]).float()}

# ---------------------------------------------------------------------------------------- #
# -------------------------------- Deep learning model ----------------------------------- #
# ---------------------------------------------------------------------------------------- #

in_size, ou_size = dataset['x_trn'].size(1), dataset['y_trn'].size(1)

h_fc = scio.loadmat('./H_fc.mat')['channel_fc']
h_fc = h_fc.transpose()
h_fc_r = np.real(h_fc)
h_fc_i = np.imag(h_fc)

ch = np.concatenate((h_fc_r, h_fc_i), axis=1)

model = LearningModel(in_size, ou_size)

# ---------------------------------------------------------------------------------------- #
# ----------------------------- Model training parameters -------------------------------- #
# ---------------------------------------------------------------------------------------- #

opt = optimizer.Adam(model.parameters(), lr=1e1)
# scheduler = optimizer.lr_scheduler.MultiStepLR(opt, [10, 100, 200], gamma=0.1, last_epoch=-1)
scheduler = optimizer.lr_scheduler.MultiStepLR(opt, [10000], gamma=0.1, last_epoch=-1)
criterion = nn.MSELoss()

batch_size = num_tr
num_of_epoch = 500

train_info = {
    'tr_loss': [],
    'val_loss': [],
    'corr': []
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# ---------------------------------------------------------------------------------------- #
# --------------------------------- Training process ------------------------------------- #
# ---------------------------------------------------------------------------------------- #

fig, ax1 = plt.subplots(1, figsize=(8, 4))
ax2 = ax1.twinx()

for epoch in range(num_of_epoch):

    batch_count = 0
    batch_per_epoch = np.ceil(np.divide(dataset['x_trn'].size(0), batch_size)).astype('int32')

    model.train()

    while batch_count < batch_per_epoch:

        start = batch_count * batch_size
        end = np.minimum(start + batch_size, dataset['x_trn'].size(0))
        batch_count += 1

        X = dataset['x_trn'][start:end, :].to(device)
        Y = dataset['y_trn'][start:end, :].to(device)
        Y_pred = model(X)
        loss = criterion(Y, Y_pred)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_info['tr_loss'].append(loss.item())

        if batch_count % 10 == 0:
            print("Epoch: {:d}/{:d}, Batch: {:d}/{:d}, Training loss: {:.5f}.".format(epoch + 1,
                                                                                      num_of_epoch,
                                                                                      batch_count,
                                                                                      batch_per_epoch,
                                                                                      loss.item()))

    # --------------------------------- Validation ------------------------------------- #
    model.eval()

    batch_size_eval = 1000
    num_of_batch_eval = np.ceil(np.divide(dataset['x_val'].size(0), batch_size_eval))
    batch_count = 0
    val_loss = 0

    while batch_count < num_of_batch_eval:
        start = batch_count * batch_size_eval
        end = np.minimum(start + batch_size_eval, dataset['x_val'].size(0))
        batch_count += 1

        X = dataset['x_val'][start:end, :].to(device)
        Y = dataset['y_val'][start:end, :].to(device)
        Y_pred = model(X)
        loss = criterion(Y, Y_pred)
        val_loss += loss.item()

    train_info['val_loss'].append(val_loss / num_of_batch_eval)

    # ---------------------------- Channel space correlation -------------------------------- #
    model_param = {
        'r': torch.Tensor.cpu(model.H_r.detach()).numpy(),
        'i': torch.Tensor.cpu(model.H_i.detach()).numpy()
    }

    h = model_param['r'] + 1j * model_param['i']
    w, v = np.linalg.eigh(h @ np.conj(h).transpose())

    h_dir = v[:, -1].reshape((v.shape[0], -1))

    train_info['corr'].append(np.abs(h_fc @ np.conj(h_dir)).squeeze().item())

    print("Epoch: {:d}/{:d}, Val loss: {:.5f}, Corr: {:.5f}".format(epoch + 1,
                                                                    num_of_epoch,
                                                                    train_info['val_loss'][-1],
                                                                    train_info['corr'][-1]))

    # --------------------------------- Plotting ------------------------------------- #
    ax1.plot(range(epoch + 1), np.array(train_info['val_loss']), '-k', alpha=1, label='Validation Loss')
    ax2.plot(range(epoch + 1), np.array(train_info['corr']), '-b', alpha=1, label='Channel correlation')
    # ax.set_xscale('log')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')

    ax2.set_ylabel('Correlation Coefficient')
    ax2.spines['right'].set_color('b')
    ax2.yaxis.label.set_color('b')
    ax2.tick_params(colors='b', which='both')

    ax1.grid(True)
    # ax1.legend()
    # ax2.legend()

    plt.draw()
    plt.pause(0.001)

    ax1.cla()
    ax2.cla()

    # ---------------------------- Learning rate adjust -------------------------------- #
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    if current_lr != new_lr:
        print("Learning rate reduced to {0:.5f}.".format(new_lr))

torch.save(model.state_dict(), 'trained_params_' + str(num_tr) + '_3bit.pt')

scio.savemat('train_info_' + str(num_tr) + '_3bit.mat', {'train_info': train_info})

H_r = torch.Tensor.cpu(model.H_r.detach()).numpy()
H_i = torch.Tensor.cpu(model.H_i.detach()).numpy()

scio.savemat('critic_params_trsize_' + str(num_tr) + '_epoch_' + str(num_of_epoch) + '_3bit.mat', {'H_r': H_r, 'H_i': H_i})
