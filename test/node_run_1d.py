# Inspired from: https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130

import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import nolds
import torchdiffeq
import json
import datetime

sys.path.append('..')
from data.preprocess import *
from model.ode import *
from util.util import *






def le_train(model, device, dataset, optimizer, criterion, epochs, time_step, y_multistep):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.reshape(-1, 1).to(device), Y.reshape(-1, 1).to(device), X_test.reshape(-1, 1).to(device), Y_test.reshape(-1, 1).to(device)
    N = Y.shape[0]
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()            


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

        ##### test multi_step #####
        if i == (epochs-1):
            err, Y, pred = multi_step_pred_err(model, X_test, device, time_step, N)

            pdf_path = "multi_pred_node.pdf"
            plot_multi_traj(X_test.detach().cpu(), pred, pdf_path)
            print("multi-step err:", np.linalg.norm(err))

        print(i, train_loss.item(), test_loss)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist





if __name__ == '__main__':

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epochs = 1000 # 1000 epochs
    lr = 1e-3
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    time_step = 1
    t = torch.linspace(0, time_step, 2)
    dim = 1 #5 #10, 30
    emb_d = 4
    matrix_d = 2
    partition = 0.7


    ###### 1. Data ######
    # input to LSTM: (N, L, H_in): N = batch size, L = sequence length, H_in: input feature size

    data = load_longer_data()
    X_s, y_s = standardize(data)
    # X, y = split_sequences(X_s, y_s, dim, dim)
    # print("after split", X.shape, y.shape)
    
    N = X_s.shape[0]
    # X_train, X_test, y_train, y_test = train_test_split(N, X_s, y_s)
    X_train, X_test, y_train, y_test = train_test_split_1d(N, X_s, y_s, partition)

    plot_data(X_train, X_test, y_train, y_test, partition, N)


    # convert to pytorch tensor
    X_train = torch.squeeze(torch.Tensor(X_train)).double()
    X_test = torch.squeeze(torch.Tensor(X_test)).double()
    y_train = torch.squeeze(torch.Tensor(y_train)).double()
    y_test = torch.squeeze(torch.Tensor(y_test)).double()
    dataset = [X_train, y_train, X_test, y_test]

    '''timestamp = datetime.datetime.now()
    with open(str(timestamp)+'.txt', 'w') as f:
        entry = {'X_train': X_train.tolist(), 'y_train': y_train.tolist(), 'X_test': X_test.tolist(), 'y_test': y_test.tolist()}
        json.dump(entry, f)'''

    print("X_train:", X_train.shape, "\n", "X_test:", X_test.shape, "\n", "y_train:", y_train.shape, "\n", "y_test:", y_test.shape)



    ##### 2. Compute LE #####
    True_LE = nolds.lyap_r(data)
    print("LE", True_LE)



    ###### 3. Train ######
    model = ODE(dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train
    pred_train, true_train, pred_test, loss_hist, test_loss_hist = le_train(model, device, dataset, optimizer, criterion, n_epochs, time_step, X_test)


    ###### 4. Save ######
    # Save Trained Model
    model_path = "../test_result/node_model.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved new model!")

    # Save whole trajectory
    # print(pred_train)
    np.savetxt('../test_result/'+"node_pred_traj.csv", np.asarray(pred_train).reshape(-1, 1), delimiter=",")

    np.savetxt('../test_result/'+"node_true_traj.csv", np.asarray(true_train).reshape(-1, 1), delimiter=",")

    loss_hist = torch.stack(loss_hist)
    np.savetxt('../test_result/'+"node_train_loss.csv", np.asarray(loss_hist.detach().cpu()), delimiter=",")

    np.savetxt('../test_result/'+"node_test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    # ac_loss_hist = torch.stack(ac_loss_hist)
    # np.savetxt('../test_result/'+"node_autocorr_loss.csv", np.asarray(ac_loss_hist.detach().cpu()), delimiter=",")

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
    dataY_plot = py.numpy().reshape(-1, 1)

    data_predict = mm.inverse_transform(data_predict) # reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    # print("dataY", dataY_plot.shape)
    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])

    # Plot
    fig, ax = subplots(figsize=(24,8))
    train_test_cutoff = round(0.70 * N)
    axvline(x=train_test_cutoff, c='b', linestyle='--') # size of the training set

    ax.plot(true, label='Actual Data', color=(0.25, 0.25, 0.25), marker='o', linewidth=5, alpha=0.8)
    ax.plot(preds, label='Predicted Data', color="slateblue", marker='o', linewidth=5, alpha=0.8)

    ax.grid(True)
    # ax.set_xlabel(r"$\tau$", fontsize=40)
    # ax.set_ylabel(r"$C_{x,x}(\tau)$", fontsize=40)
    ax.tick_params(labelsize=30)
    tight_layout()
    legend()
    pdf_path = "whole_plot_node.png"
    fig.savefig(pdf_path, format='png', dpi=400)

    LE_learned = nolds.lyap_r(preds)
    print("LE Learned", LE_learned)

    LE_t = nolds.lyap_r(true)
    print("LE True", LE_t)

    LE_td = nolds.lyap_r(true)
    print("LE True diff dy_n", LE_td)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_learned - LE_t))

    LE_unseen_p = nolds.lyap_r(preds[train_test_cutoff:])
    print("LE Learned unseen", LE_unseen_p)

    LE_unseen_t = nolds.lyap_r(true[train_test_cutoff:])
    print("LE Learned true", LE_unseen_t)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_unseen_p - LE_unseen_t))
