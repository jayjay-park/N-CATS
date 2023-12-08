import pandas as pd
import numpy as np
from matplotlib.pyplot import *
import torch
import torch.nn as nn
import torchdiffeq
from mycolorpy import colorlist as mcp
import sys

sys.path.append('..')
from data.preprocess import *


# Time Integrator
def solve_odefunc(odefunc, t, y0):
    ''' Solve odefunction using torchdiffeq.odeint() '''

    solution = torchdiffeq.odeint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]
    return final_state

# Time Integrator
def solve_sdefunc(odefunc, t, y0):
    ''' Solve odefunction using torchsde.sdeint() '''

    solution = torchsde.sdeint(odefunc, y0, t, rtol=1e-9, atol=1e-9)
    final_state = solution[-1]
    return final_state



def loss_func_vae(y, y_hat, mean, log_var, device, Y_log_var, true_corr):
    reproduction_loss = torch.nn.MSELoss(reduction='mean')(y, y_hat)

    # corr = torch.zeros(100).to(device)
    # window_size = 50
    # base_line = y_hat[0:window_size]
    
    # for i in range(0, 100):
    #     corr[i] = _corr(base_line.to(device), y_hat[i+window_size:i+window_size*2])
    # corr = corr - torch.mean(y_hat)**2


    # print(corr)
    # autocorr_loss = (1 - corr)
    # autocorr_loss = _corr(log_var, Y_log_var)

    # tau_x, pred_corr = auto_corr(y_hat, device)
    # auto_corr_loss = torch.norm(true_corr - pred_corr.to(device))
    # train_loss = MSE_loss + reg_param * auto_corr_loss

    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


######### Evaluate ##########


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


def evaluate_latent(model, time_step, X_test, Y_test, device, criterion):

  with torch.no_grad():
    model.eval()

    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    # y_pred_test = solve_odefunc(model, t_eval_point, X_test).to(device)
    y_pred_test = model(X_test)

    # save predicted node feature for analysis
    pred_test = y_pred_test.detach().cpu()
    Y_test = Y_test.detach().cpu()

    test_loss = criterion(pred_test, Y_test).item()

  return pred_test, test_loss



def evaluate_nvade(model, time_step, X_test, Y_test, device, criterion, true_corr):

  with torch.no_grad():
    model.eval()

    y_hat, mean, log_var = model(X_test)
    Y_y_hat, Y_mean, Y_log_var = model(Y_test)

    # save predicted node feature for analysis
    pred_test = y_hat.detach().cpu()
    # Y_test = Y_test.detach().cpu()

    test_loss = loss_func_vae(Y_test, y_hat, mean, log_var, device, Y_log_var, true_corr).item()
    # loss_func_vae(y, y_hat, mean, log_var, device, Y_log_var)
  return pred_test, test_loss



########## MULTI STEP ###########


def multi_step_pred_err(model, true_traj, device, time_step, integration_time):

    # num_of_extrapolation_dataset
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    true_traj = torch.tensor(true_traj)
    num_data = true_traj.shape[0]
    dim = 1
    test_t = torch.linspace(0, integration_time, num_data)
    pred_traj = torch.zeros(num_data, dim).to(device)

    with torch.no_grad():
        model.eval()
        model.double()
        model.to(device)

        # initialize X
        print(true_traj[0])
        X = true_traj[0].to(device)

        # calculating outputs
        for i in range(num_data):
            pred_traj[i] = X # shape [3]
            # cur_pred = model(t.to(device), X.double())
            cur_pred = solve_odefunc(model, t_eval_point, X.double()).to(device)
            X = cur_pred

        # save predicted trajectory
        Y = true_traj.detach().cpu()
        pred = pred_traj.detach().cpu()
        # pred_traj_csv = np.asarray(pred)
        # true_traj_csv = np.asarray(Y)

        err = np.abs(pred[:] - Y[:])

    return err, Y, pred


def multi_step_pred_err_latent(model, true_traj, device, time_step):

    # num_of_extrapolation_dataset
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    # true_traj = torch.tensor(true_traj)
    num_data = true_traj.shape[0]
    dim = 1
    test_t = torch.linspace(0, num_data, num_data)
    pred_traj = torch.zeros(num_data, dim).to(device)

    with torch.no_grad():
        model.eval()
        model.double()
        model.to(device)

        # initialize X
        # if true_traj.shape[0] == 1:
        X = true_traj[0].to(device)
        # else:
        #     X = true_traj[0].to(device)
        #     init_X = true_traj[0][0].to(device)

        # calculating outputs
        for i in range(num_data):
            print(i, X)
            pred_traj[i] = X[0] # shape [3]
            X = X.reshape(-1, 1)
            cur_pred = model(X).to(device)
            X = cur_pred


        # save predicted trajectory
        Y = true_traj.detach().cpu()
        pred = pred_traj.detach().cpu()
        # pred_traj_csv = np.asarray(pred)
        # true_traj_csv = np.asarray(Y)

        err = np.abs(pred[:] - Y[:])

    return err, Y, pred


def multi_step_pred_err_lstm(model, true_traj, device, time_step):

    # num_of_extrapolation_dataset
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()
    # true_traj = torch.tensor(true_traj)
    num_data = true_traj.shape[0]
    dim = 10
    test_t = torch.linspace(0, num_data, num_data)
    pred_traj = torch.zeros((num_data, 1))

    y_split, y_split_2 = split_sequences(true_traj, true_traj, dim, dim)
    y = torch.tensor(y_split)

    with torch.no_grad():
        model.eval()
        model.double()

        # initialize X
        X = y[0]
        print(isinstance(X, np.ndarray))

        # calculating outputs
        for i in range(num_data):
            print(X.shape)
            pred_traj[i, :] = X

            X = X.reshape(-1, 1)
            cur_pred = model(X)
            X = cur_pred

        rolled_out_pred = batch_to_full(pred_traj, device)

        # save predicted trajectory
        Y = true_traj
        pred = rolled_out_pred
        # pred_traj_csv = np.asarray(pred)
        # true_traj_csv = np.asarray(Y)

        err = np.abs(pred[:] - Y[:])

    return err, Y, pred



def batch_to_full(y_pred, device):
    '''For both 1 dim or N dim dataset'''

    N = y_pred.shape[0]
    one_shot_y = torch.zeros(y_pred.shape[0]).to(device) 

    for j in range(N):
        one_shot_y[j] = (y_pred[j][0])
    one_shot_y = one_shot_y.reshape(-1, 1)

    return one_shot_y


####### PLOT ########

def plot_data(X_train, X_test, y_train, y_test, partition, N):
    # Plot Dataset
    figure(figsize=(15,6))
    train_test_cutoff = round(partition * N)
    axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set
    print("N", N) #3355
    x = np.linspace(0, N, N)
 
    color1=mcp.gen_color(cmap="winter",n=5)

    plot(x[:train_test_cutoff-1], X_train, label='X_train', alpha=0.8, linewidth=1, color=color1[0]) # actual plot
    plot(x[1:train_test_cutoff], y_train, label='y_train', alpha=0.8, marker='o', linewidth=1, color=color1[1])
    plot(x[train_test_cutoff:-1], X_test, label='X_test', linewidth=1, alpha=0.7, color=color1[2]) # predicted plot
    plot(x[train_test_cutoff+1:], y_test, label='y_test', linewidth=1, alpha=0.7, marker='o', color=color1[3])
    title('Dataset')
    legend()
    tight_layout()
    savefig("dataset.png", dpi=300)
    return


def plot_multi_traj(Y, pred, pdf_path):

    # savefig
    fig, ax = subplots(figsize=(36,12))
    ax.plot(np.array(Y), color=(0.25, 0.25, 0.25), marker='o', linewidth=5, alpha=0.8)
    ax.plot(np.array(pred), color="slateblue", marker='o', linewidth=5, alpha=0.8)
    ax.grid(True)
    ax.set_xlabel(r"$t$", fontsize=36)
    # ax.set_ylabel(r"$Price$", fontsize=36)
    ax.tick_params(labelsize=36)
    tight_layout()

    fig.savefig(pdf_path, format='pdf', dpi=400)
    close()
    return

########## Autocorr ############

def _corr(x, y):
    print(x.shape)
    x = x.view(-1)
    print(x.shape)
    y = y.view(-1)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x.expand_as(x))
    ym = y.sub(mean_y.expand_as(y))
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val



def auto_corr(y_hat, device):
    ''' computing auto_corr in batch 
        input = 1D time series 
        output = scalar '''

    window_size = 500 #50
    step = 1
    tau = y_hat.shape[0] - window_size *2 #100
    # corr = torch.zeros(tau).to(device)
    list_i = torch.arange(0, tau+1, step) # 0 ... tau

    base_line = y_hat[0:window_size]
    subseq = torch.stack([y_hat[i:i+window_size] for i in list_i]).to(device)

    def corr_val(subseq):
        return torch.dot(torch.flatten(base_line).to(device), torch.flatten(subseq))
        # return _corr(base_line.to(device), subseq)

    # Use torch.vmap
    batch_corr_val = torch.vmap(corr_val)
    corr = batch_corr_val(subseq).to(device)
    print(corr.shape)

    corr = corr/window_size - torch.mean(y_hat)**2
    tau_x = torch.arange(0, tau+1, step)

    return tau_x, corr



def plot_ac(tau_x, corr_list, pdf_path):
    # savefig
    fig, ax = subplots(figsize=(36,12))
    ax.plot(np.array(tau_x.detach().cpu()), np.array(corr_list.detach().cpu()), color="slateblue", marker='o', linewidth=5, alpha=0.8)
    ax.grid(True)
    ax.set_xlabel(r"$\tau$", fontsize=48)
    ax.set_ylabel(r"$C_{x,x}(\tau)$", fontsize=48)
    ax.tick_params(labelsize=48)
    tight_layout()

    fig.savefig(pdf_path, format='pdf', dpi=400)
    close()
    return


def plot_loss_MSE(MSE_train, MSE_test, AC_test):
    fig, axs = subplots(1, figsize=(24, 12)) #, sharey=True
    fig.suptitle("Loss Behavior of MSE Loss", fontsize=24)

    colors = cm.tab20b(np.linspace(0, 1, 20))

    # Training Loss
    x = np.arange(0, MSE_train.shape[0])

    axs.plot(x, MSE_train, c=colors[15], label='Train Loss', alpha=0.9, linewidth=5)
    axs.plot(x, MSE_test, c=colors[1], label='Test Loss', alpha=0.9, linewidth=5)
    axs.plot(x, AC_test, c=colors[5], label='Auto Corr Loss', alpha=0.9, linewidth=5)
    axs.grid(True)
    axs.legend(loc='best', fontsize=40)
    # axs.set_ylabel(r'$\mathcal{L}$', fontsize=24)
    axs.set_xlabel('Epochs', fontsize=40)
    axs.tick_params(labelsize=40)

    tight_layout()
    savefig('loss_ncats.png', format='png', dpi=600, bbox_inches ='tight', pad_inches = 0.1)

    return

if __name__ == '__main__':
    MSE_train = np.genfromtxt("../test_result/nsde/ncats_sde_train_loss.csv", delimiter=",", dtype=float)
    MSE_test = np.genfromtxt("../test_result/nsde/ncats_test_loss.csv", delimiter=",", dtype=float)
    AC_test = np.genfromtxt("../test_result/nsde/ncats_sde_autocorr_loss.csv", delimiter=",", dtype=float)
    
    plot_loss_MSE(MSE_train[10:], MSE_test[10:], AC_test[10:])