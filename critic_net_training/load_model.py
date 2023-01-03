import torch
import numpy as np
import scipy.io as scio

from build_model import LearningModel


if __name__ == '__main__':

    in_size = 512
    ou_size = 1

    ch = np.random.randn(3, 4)

    num_tr = 1000
    Nbit = 6

    model = LearningModel(in_size, ou_size, ch)
    model.load_state_dict(torch.load('trained_params_' + str(num_tr) + '_' + str(Nbit) + 'bit.pt'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    H_r = torch.Tensor.cpu(model.H_r.detach()).numpy()
    H_i = torch.Tensor.cpu(model.H_i.detach()).numpy()

    scio.savemat('H_est_' + str(num_tr) + '_' + str(Nbit) + 'bit.mat', {'H_r': H_r, 'H_i': H_i})

    model.eval()

    # ---------------------------------------------------------------------------------------- #
    # ---------------------------------- Just an example ------------------------------------- #
    # ---------------------------------------------------------------------------------------- #

    num_of_test_point = 10000
    x_test = torch.randn((num_of_test_point, in_size)).float()  # replace this with yours
    y_pred = torch.empty((num_of_test_point, ou_size))

    batch_size = 100
    num_of_batch = np.ceil(np.divide(x_test.size(0), batch_size))

    batch_count = 0
    while batch_count < num_of_batch:

        start = batch_count * batch_size
        end = np.minimum(start + batch_size, x_test.size(0))
        batch_count += 1

        X = x_test[start:end, :].to(device)
        Y_pred = model(X)
        y_pred[start:end, :] = Y_pred

    Y_pred_np = torch.Tensor.cpu(y_pred.detach()).numpy()
    # scio.savemat('your_pred_file.mat', {'y_pred': Y_pred_np})

    print("Finished.\n")
