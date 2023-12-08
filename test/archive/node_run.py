# Inspired from: https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import nolds
import torchdiffeq

sys.path.append('..')
from data.preprocess import *
from model.ode import *



    
# Time Integrator
def solve_odefunc(odefunc, t, y0):
    ''' Solve odefunction using torchdiffeq.odeint() '''

    solution = torchdiffeq.odeint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]
    return final_state


def le_train(model, device, dataset, optimizer, criterion, epochs, time_step):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()            
    reg_param = 0.02


    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        y_pred = solve_odefunc(model.to(device), t_eval_point, X).to(device)

        optimizer.zero_grad()
        # MSE Output Loss
        MSE_loss = criterion(y_pred, Y)
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
    y_pred_test = solve_odefunc(model, t_eval_point, X_test).to(device)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y_test = Y_test.detach().cpu()

    test_loss = criterion(pred_test, Y_test).item()

  return pred_test, test_loss




if __name__ == '__main__':

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epochs = 1000 # 1000 epochs
    lr = 1e-3
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    time_step = 1
    t = torch.linspace(0, time_step, 2)
    dim = 5 #5 #10, 30
    emb_d = 4
    matrix_d = 2


    ###### 1. Data ######
    # input to LSTM: (N, L, H_in): N = batch size, L = sequence length, H_in: input feature size

    data = load_data()
    X_s, y_s = standardize(data)
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
    True_LE = nolds.lyap_e(data, emb_dim=emb_d, matrix_dim=matrix_d, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE", True_LE)



    ###### 3. Train ######
    model = ODE(dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train
    pred_train, true_train, pred_test, loss_hist, test_loss_hist = le_train(model, device, dataset, optimizer, criterion, n_epochs, time_step)


    # Plot
    # split the sequence
    X, y = data, data
    mm = MinMaxScaler()
    #ss = StandardScaler()
    X = mm.fit_transform(X.reshape(-1, 1))
    y = mm.fit_transform(y.reshape(-1, 1))

    px, py = split_sequences(X, y, dim, dim) 
    # converting to tensors
    px, py = torch.squeeze(torch.Tensor(px)), torch.squeeze(torch.Tensor(py))
    # reshaping the dataset
    update_px = torch.reshape(px, (px.shape[0], dim))
    print(update_px.shape) #torch.Size([3337, 10])

    train_predict = solve_odefunc(model, t.to(device), update_px.to(device).double())
    data_predict = train_predict.detach().cpu().numpy() # numpy conversion
    dataY_plot = py.numpy()

    data_predict = mm.inverse_transform(data_predict) # reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    # print("dataY", dataY_plot.shape)
    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])

    # Plot
    figure(figsize=(10,6))
    train_test_cutoff = round(0.60 * N)
    axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set

    plot(true, label='Actual Data') # actual plot
    plot(preds, label='Predicted Data') # predicted plot
    title('Time-Series Prediction')
    legend()
    savefig("ode_whole_plot.png", dpi=300)

    LE_learned = nolds.lyap_e(preds, emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE Learned", LE_learned)

    LE_t = nolds.lyap_e(true, emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE True", LE_t)

    LE_td = nolds.lyap_e(true, emb_dim=4, matrix_dim=2, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE True diff dy_n", LE_td)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_learned - LE_t))

    LE_unseen_p = nolds.lyap_e(preds[train_test_cutoff:], emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE Learned unseen", LE_unseen_p)

    LE_unseen_t = nolds.lyap_e(true[train_test_cutoff:], emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE Learned true", LE_unseen_t)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_unseen_p - LE_unseen_t))
