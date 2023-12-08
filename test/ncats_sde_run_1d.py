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
from model.sde_latent import *
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
    reg_param = 8e-3 #0.02 #0.04 #0.1


    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        # y_pred = solve_odefunc(model, t_eval_point, X).to(device)
        y_pred = model(X).to(device)
        optimizer.zero_grad()
        
        # MSE Output Loss
        MSE_loss = criterion(y_pred, Y)

        # Auto-corr Diff Loss
        tau_x, pred_corr = auto_corr(y_pred, device)
        auto_corr_loss = torch.norm(true_corr - pred_corr.to(device))
        train_loss = MSE_loss + reg_param * auto_corr_loss
        # train_loss = MSE_loss
        
        train_loss.backward()
        optimizer.step()

        # leave it for debug purpose for now, and remove it
        pred_train.append(y_pred.detach().cpu().numpy())
        true_train.append(Y.detach().cpu().numpy())

        loss_hist.append(train_loss)
        ac_loss_hist.append(auto_corr_loss)

        ##### test one_step #####
        pred_test, test_loss = evaluate_latent(model, time_step, X_test, Y_test, device, criterion)
        test_loss_hist.append(test_loss)

        # auto_corr_loss = 0
        print(i, train_loss.item(), MSE_loss.item(), auto_corr_loss.item(), test_loss)

        ##### test multi_step #####
        '''if i % 1000 == 0:
            y_s = y_s[:70].reshape(-1, 1)
            err, y_msp, pred = multi_step_pred_err_latent(model, y_s, device, time_step)

            pdf_path = "ncats_msp_nd_" + str(i) +".pdf"
            err_path = "ncats_mse_nd_" + str(i) +".pdf"
            plot_multi_traj(y_s, pred, pdf_path)
            plot_multi_traj(err, err, err_path)
            print("multi-step err:", np.linalg.norm(err))'''

        if i == (epochs-1):
            y_s = y_s[:1500].reshape(-1, 1)
            err, y_msp, pred = multi_step_pred_err_latent(model, y_s, device, time_step)

            pdf_path = "ncats_msp_nd_" + str(i) +".pdf"
            err_path = "ncats_mse_nd_" + str(i) +".pdf"
            plot_multi_traj(y_s, pred, pdf_path)
            plot_multi_traj(err, err, err_path)

            print("multi-step err:", np.linalg.norm(err))

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, ac_loss_hist





if __name__ == '__main__':

    torch.manual_seed(42)
    torch.set_printoptions(precision=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epochs = 800# 1000 epochs
    lr = 5e-4
    criterion = torch.nn.MSELoss() 
    time_step = 1 #1
    t = torch.linspace(0, time_step, 2)
    dim = 1 #5
    emb_d = 4
    matrix_d = 2
    partition = 0.7
    latent_dim = 16


    ###### 1. Data ######
    # input to LSTM: (N, L, H_in): N = batch size, L = sequence length, H_in: input feature size

    data = load_longer_data()
    X_s, y_s = standardize(data)

    N = X_s.shape[0]
    X_train, X_test, y_train, y_test = train_test_split_1d(N, X_s, y_s, partition)

    plot_data(X_train, X_test, y_train, y_test, partition, N)

    # for LE loss
    train_test_cutoff = round(partition * N)
    full_y_train = torch.tensor(y_s[:train_test_cutoff])
    print("y1", full_y_train.shape)

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
    torch.cuda.empty_cache()
    sde = SDE(latent_dim)
    model = latent_SDE(dim=dim, latent_dim=latent_dim, SDE=sde.to(device), t=t.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    pred_train, true_train, pred_test, loss_hist, test_loss_hist, ac_loss_hist = le_train(model, device, dataset, optimizer, criterion, n_epochs, time_step, matrix_d, corr_list, X_test)


    ###### 4. Save ######
    # Save Trained Model
    model_path = "../test_result/ncats_sde_model.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved new model!")

    # Save whole trajectory
    np.savetxt('../test_result/'+"ncats_sde_pred_traj.csv", np.asarray(pred_train).reshape(-1, 1), delimiter=",")

    np.savetxt('../test_result/'+"ncats_sde_true_traj.csv", np.asarray(true_train).reshape(-1, 1), delimiter=",")

    loss_hist = torch.stack(loss_hist)
    np.savetxt('../test_result/'+"ncats_sde_train_loss.csv", np.asarray(loss_hist.detach().cpu()), delimiter=",")

    np.savetxt('../test_result/'+"ncats_sde_test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    ac_loss_hist = torch.stack(ac_loss_hist)
    np.savetxt('../test_result/'+"ncats_sde_autocorr_loss.csv", np.asarray(ac_loss_hist.detach().cpu()), delimiter=",")


    '''# simulate Neural ODE
    model_name = "nsde/ncats_sde_model.pt"
    model_path = "../test_result/"+str(model_name)
    # pdf_path = '../plot/corr_'+str(model_name)+'_'+str(torch.round(init, decimals=4).tolist())+'.pdf'
    # Load the saved model
    sde = SDE(latent_dim)
    model = latent_SDE(dim=dim, latent_dim=latent_dim, SDE=sde.to(device), t=t.to(device))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Finished Loading model")

    # Assuming 'model' is your PyTorch model
    model_parameters = model.parameters()
    # Get the device of the first parameter
    first_param_device = next(model_parameters).device
    print(f"Model is on device: {first_param_device}")'''


    ###### 5. Plot ######
    # split the sequence

    X, y = data, data
    mm = MinMaxScaler()
    # ss = StandardScaler()
    X = mm.fit_transform(X.reshape(-1, 1))
    y = mm.fit_transform(y.reshape(-1, 1))

    px, py = split_sequences(X, y, dim, dim)
    # converting to tensors
    px, py = torch.squeeze(torch.Tensor(px)), torch.squeeze(torch.Tensor(py))
    # reshaping the dataset
    update_px = torch.reshape(px, (px.shape[0], dim)).double().to(device)
    print(update_px.shape)
    print(f"update_px device: {update_px.device}")
    model = model.double().to(device)

    torch.cuda.empty_cache()
    num=1600
    train_predict = model(update_px[:num]).to(device)
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
    fig, ax = subplots(figsize=(24,8))
    # axvline(x=train_test_cutoff, c='b', linestyle='--') # size of the training set

    ax.plot(true[:num], label='Actual Data', color=(0.25, 0.25, 0.25), marker='o', linewidth=5, alpha=0.8)
    ax.plot(preds, label='Predicted Data', color="slateblue", marker='o', linewidth=5, alpha=0.8)

    ax.grid(True)
    # ax.set_xlabel(r"$\tau$", fontsize=40)
    # ax.set_ylabel(r"$C_{x,x}(\tau)$", fontsize=40)
    ax.tick_params(labelsize=30)
    tight_layout()
    legend()
    pdf_path = "whole_plot_ncats_sde.png"
    fig.savefig(pdf_path, format='png', dpi=400)
    close()

    y_s = X_test.reshape(-1, 1)
    err, y_msp, pred = multi_step_pred_err_latent(model, y_s, device, time_step)
    i=1000
    pdf_path = "ncats_msp_nd_" + str(i) +".pdf"
    err_path = "ncats_mse_nd_" + str(i) +".pdf"
    plot_multi_traj(y_s, pred, pdf_path)
    plot_multi_traj(err, err, err_path)

    print("multi-step err:", np.linalg.norm(err))




    LE_learned = nolds.lyap_r(preds)
    print("LE Learned", LE_learned)

    LE_t = nolds.lyap_r(true)
    print("LE True", LE_t)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_learned - LE_t))

    print("size", preds[train_test_cutoff:])
    LE_unseen_p = nolds.lyap_r(preds[train_test_cutoff:])
    print("LE Learned unseen", LE_unseen_p)

    LE_unseen_t = nolds.lyap_r(true[train_test_cutoff:])
    print("LE Learned true", LE_unseen_t)

    print("Whole Traj Norm Diff:", np.linalg.norm(LE_unseen_p - LE_unseen_t))