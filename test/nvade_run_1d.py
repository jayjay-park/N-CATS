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
from model.nvade import *
from model.ode import *
from util.util import *







def le_train(model, device, dataset, optimizer, criterion, epochs, time_step, matrix_d, true_corr, y_s):
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=6)

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist, ac_loss_hist = ([] for i in range(5))
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.reshape(-1, 1).to(device), Y.reshape(-1, 1).to(device), X_test.reshape(-1, 1).to(device), Y_test.reshape(-1, 1).to(device)
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()    
    # one_shot_y = torch.zeros(X.shape[0]).to(device)  
    true_corr = true_corr.to(device)      
    model.to(device)
    true_encoded_log_val = torch.var(Y).to(device)
    reg_param = 1e-4 #0.02 #0.04 #0.1


    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        y_hat, mean, log_var = model(X)
        # for loss computation
        Y_y_hat, Y_mean, Y_log_var = model(Y)
        # torch.Size([7546, 1]) torch.Size([7546, 2]) torch.Size([7546, 2])

        optimizer.zero_grad()
        train_loss = loss_func_vae(Y, y_hat, mean, log_var, device, Y_log_var, true_corr)
        
        train_loss.backward()
        optimizer.step()

        # leave it for debug purpose for now, and remove it
        pred_train.append(y_hat.detach().cpu().numpy())
        true_train.append(Y.detach().cpu().numpy())

        loss_hist.append(train_loss)
        # ac_loss_hist.append(auto_corr_loss)

        ##### test one_step #####
        pred_test, test_loss = evaluate_nvade(model, time_step, X_test, Y_test, device, criterion, true_corr)
        test_loss_hist.append(test_loss)

        # auto_corr_loss = 0
        print(i, train_loss.item(), test_loss)

        ##### test multi_step #####
        if i == (epochs-1):
            err, Y, pred = multi_step_pred_err_nvade(model, y_s, device, time_step)

            pdf_path = "multi_pred_nvade.pdf"
            plot_multi_traj(y_s, pred, pdf_path)

            print("multi-step err:", np.linalg.norm(err))

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, ac_loss_hist






if __name__ == '__main__':

    torch.manual_seed(42)
    torch.set_printoptions(precision=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epochs = 500 # 1000 epochs
    lr = 5e-4
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    time_step = 1 #1
    t = torch.linspace(0, time_step, 2)
    dim = 1 #5
    emb_d = 4
    matrix_d = 2
    partition = 0.7


    ###### 1. Data ######
    # input to LSTM: (N, L, H_in): N = batch size, L = sequence length, H_in: input feature size

    data = load_longer_data()
    X_s, y_s = standardize(data)

    N = X_s.shape[0]
    X_train, X_test, y_train, y_test = train_test_split_1d(N, X_s, y_s, partition)

    plot_data(X_train, X_test, y_train, y_test, partition, N)

    # for LE loss
    # train_test_cutoff = round(partition * N)
    # full_y_train = torch.tensor(y_s[:train_test_cutoff])
    # print("y1", full_y_train.shape)

    # convert to pytorch tensor
    X_train = torch.squeeze(torch.Tensor(X_train)).double()
    X_test = torch.squeeze(torch.Tensor(X_test)).double()
    y_train = torch.squeeze(torch.Tensor(y_train)).double()
    y_test = torch.squeeze(torch.Tensor(y_test)).double()

    dataset = [X_train, y_train, X_test, y_test]

    print("X_train:", X_train.shape, "\n", "X_test:", X_test.shape, "\n", "y_train:", y_train.shape, "\n", "y_test:", y_test.shape)



    ##### 2. Compute Auto-Correlation #####
    tau_x, corr_list = auto_corr(y_train, device)
    tau_whole_x, corr_whole_list = auto_corr(torch.tensor(X_s), device)

    pdf_path = "corr_train.pdf"
    pdf_whole_path = "corr_data.pdf"

    plot_ac(tau_x, corr_list, pdf_path)
    plot_ac(tau_whole_x, corr_whole_list, pdf_whole_path)


    ###### 3. Train ######
    encoded_dim = 4
    model = nvade(t=t.to(device), device=device, input_dim=1, hidden_dim=256, latent_dim=512, encoded_dim = encoded_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    pred_train, true_train, pred_test, loss_hist, test_loss_hist, ac_loss_hist = le_train(model, device, dataset, optimizer, criterion, n_epochs, time_step, matrix_d, corr_list, X_train)


    ###### 4. Save ######



    ###### 5. Plot ######
    # split the sequence
    X, y = data, data
    mm = MinMaxScaler()
    ss = StandardScaler()
    X = mm.fit_transform(X.reshape(-1, 1))
    y = mm.fit_transform(y.reshape(-1, 1))

    px, py = split_sequences(X, y, dim, dim)
    # converting to tensors
    px, py = torch.squeeze(torch.Tensor(px)), torch.squeeze(torch.Tensor(py))
    # reshaping the dataset
    update_px = torch.reshape(px, (px.shape[0], dim))
    print(update_px.shape)

    train_predict, mean, log_var = model(update_px.to(device).double())
    data_predict = train_predict.detach().cpu().numpy() # numpy conversion
    dataY_plot = py.numpy().reshape(-1, 1)

    data_predict = mm.inverse_transform(data_predict) # reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])

    # Plot
    figure(figsize=(10,6))
    train_test_cutoff = round(partition * N)
    axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set

    plot(true, label='Actual Data') # actual plot
    plot(preds, label='Predicted Data') # predicted plot
    title('Time-Series Prediction')
    legend()
    savefig("nvade_whole_plot.png", dpi=300)

    LE_learned = nolds.lyap_e(preds, emb_dim=emb_d, matrix_dim=matrix_d, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE Learned", LE_learned)

    LE_t = nolds.lyap_e(true, emb_dim=emb_d, matrix_dim=matrix_d, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE True", LE_t)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_learned - LE_t))

    LE_unseen_p = nolds.lyap_e(preds[train_test_cutoff:], emb_dim=emb_d, matrix_dim=matrix_d, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE Learned unseen", LE_unseen_p)

    LE_unseen_t = nolds.lyap_e(true[train_test_cutoff:], emb_dim=emb_d, matrix_dim=matrix_d, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=True)
    print("LE Learned true", LE_unseen_t)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_unseen_p - LE_unseen_t))
