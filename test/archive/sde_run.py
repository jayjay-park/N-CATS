# Inspired from: https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import nolds
import torchsde

sys.path.append('..')
from data.preprocess import *
from model.sde import *




# Time Integrator
def solve_sdefunc(odefunc, t, y0):
    ''' Solve odefunction using torchsde.sdeint() '''

    solution = torchsde.sdeint(model, y0, t, rtol=1e-9, atol=1e-9)
    final_state = solution[-1]
    return final_state


# def test_jac_node(x, eps, time_step, n):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     x = x.to(device)

#     # initialize
#     t_eval_point = torch.linspace(0, time_step, 2)
#     jac_rk4_fd = torch.zeros(n,n).double()

#     # Finite Differentiation using Central Difference Approximation
#     for i in range(n):
#         x_plus = x.clone()
#         x_minus = x.clone()

#         # create perturbed input
#         x_plus[i] = x_plus[i] + eps
#         x_minus[i] = x_minus[i] - eps

#         # create model output
#         m_plus = solve_odefunc(model, t_eval_point, x_plus)
#         #print("mp", m_plus)
#         m_minus = solve_odefunc(model, t_eval_point, x_minus)

#         # compute central diff
#         diff = m_plus.clone().detach() - m_minus.clone().detach()
#         final = diff/2/eps
#         if n == 1:
#             jac_rk4_fd = final
#         else:
#             jac_rk4_fd[:, i] = final

#     print("jac_rk4_fd\n", jac_rk4_fd)

#     return jac_rk4_fd



def le_train(model, device, dataset, optimizer, criterion, epochs, time_step):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()            
    reg_param = 0.2


    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        y_pred = solve_sdefunc(model.to(device), t_eval_point, X).to(device)

        optimizer.zero_grad()
        # MSE Output Loss
        MSE_loss = criterion(y_pred, Y)

        # # LE Diff Loss

        # jacrev = torch.func.jacrev(model, argnums=1)
        # compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0))
        # cur_model_J = compute_batch_jac(t_eval_point, X).to(device)
        # cur_LE = torch.linalg.eigvals(torch.matmul(cur_model_J, cur_model_J))
        # # cur_LE = nolds.lyap_e(y_pred., emb_dim=10, matrix_dim=4)
        # print("LE", cur_LE)
        # print("LE shape", cur_LE.shape)

        # jac_rk4_fd = test_jac_node(y_pred, eps=1e-8, time_step=1, n=1)
        # sq_m = torch.matmul(jac_rk4_fd, jac_rk4_fd.T)
        # le = torch.linalg.eig(sq_m)
        # print("Lyapunov Exponent:", le)

        # norm_diff_LE = torch.norm(True_LE - le)

        # # Train Loss
        # train_loss = reg_param * norm_diff_LE + MSE_loss
        # train_loss.backward()

        train_loss = MSE_loss
        
        train_loss.backward()
        optimizer.step()

        # leave it for debug purpose for now, and remove it
        pred_train.append(y_pred.detach().cpu().numpy())
        true_train.append(Y.detach().cpu().numpy())

        loss_hist.append(train_loss)

        ##### test one_step #####
        pred_test, test_loss = evaluate(model, time_step, X_test, Y_test, device, criterion)
        test_loss_hist.append(test_loss)

        print(i, MSE_loss.item(), test_loss)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist


def evaluate(model, time_step, X_test, Y_test, device, criterion):

  with torch.no_grad():
    model.eval()

    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    y_pred_test = solve_sdefunc(model, t_eval_point, X_test).to(device)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y_test = Y_test.detach().cpu()

    test_loss = criterion(pred_test, Y_test).item()

  return pred_test, test_loss




if __name__ == '__main__':

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epochs = 5000 # 1000 epochs
    lr = 1e-3
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    time_step = 1 #0.1 #1
    t = torch.linspace(0, time_step, 2)
    dim = 10


    ###### 1. Data ######

    # input to LSTM: (N, L, H_in): N = batch size, L = sequence length, H_in: input feature size
    data = load_data()
    X_s, y_s = standardize(data)
    #X, y = data.reshape(-1, 1), data.reshape(-1, 1)
    X, y = split_sequences(X_s, y_s, dim, dim)
    print("after split", X.shape, y.shape)
    N = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(N, X, y)

    # convert to pytorch tensor
    X_train = torch.squeeze(torch.Tensor(X_train)).double()
    X_test = torch.squeeze(torch.Tensor(X_test)).double()
    y_train = torch.squeeze(torch.Tensor(y_train)).double()
    y_test = torch.squeeze(torch.Tensor(y_test)).double()

    dataset = [X_train, y_train, X_test, y_test]

    print("X_train:", X_train.shape, "\n", "X_test:", X_test.shape, "\n", "y_train:", y_train.shape, "\n", "y_test:", y_test.shape)



    ##### 2. Compute LE #####

    True_LE = nolds.lyap_e(data, emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE", True_LE)


    ###### 3. Train ######
    # state_size = 100
    # batch_size = X_train.shape[0]

    model = SDE(dim, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train
    pred_train, true_train, pred_test, loss_hist, test_loss_hist = le_train(model, device, dataset, optimizer, criterion, n_epochs, time_step)

    # Plot
    # split the sequence
    X, y = data, data
    mm = MinMaxScaler()
    ss = StandardScaler()
    X = ss.fit_transform(X.reshape(-1, 1))
    y = mm.fit_transform(y.reshape(-1, 1))

    px, py = split_sequences(X, y, dim, dim)
    # converting to tensors
    px, py = torch.squeeze(torch.Tensor(px)), torch.squeeze(torch.Tensor(py))
    # reshaping the dataset
    update_px = torch.reshape(px, (px.shape[0], dim))
    print(update_px.shape)

    train_predict = solve_sdefunc(model, t.to(device), update_px.to(device).double())
    data_predict = train_predict.detach().cpu().numpy() # numpy conversion
    dataY_plot = py.numpy()

    data_predict = mm.inverse_transform(data_predict) # reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])

    # Plot
    figure(figsize=(10,6))
    train_test_cutoff = round(0.80 * N)
    axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set

    plot(true, label='Actual Data') # actual plot
    plot(preds, label='Predicted Data') # predicted plot
    title('Time-Series Prediction')
    legend()
    savefig("sde_whole_plot.png", dpi=300)


