# Inspired from: https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130
# LSTM
import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import warnings
import nolds
import os

sys.path.append('..')
from data.preprocess import *
from model.lstm import *
from util.util import *


def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train, X_test, y_test):

    loss_list = []
    test_loss_list = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if (epoch % 100) == 0 or (epoch == n_epochs-1):
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_loss.item())) 
        loss_list.append(loss.item())
        test_loss_list.append(test_loss.item())

                ##### test multi_step #####
        '''if i % 1000 == 0:
            y_s = y_s[:70].reshape(-1, 1)
            err, y_msp, pred = multi_step_pred_err_latent(model, y_s, device, time_step)

            pdf_path = "ncats_msp_nd_" + str(i) +".pdf"
            err_path = "ncats_mse_nd_" + str(i) +".pdf"
            plot_multi_traj(y_s, pred, pdf_path)
            plot_multi_traj(err, err, err_path)
            print("multi-step err:", np.linalg.norm(err))'''

        if epoch == (n_epochs-1):
            # y_s = y.reshape(-1, 1)
            err, y_msp, pred = multi_step_pred_err_lstm(lstm, X_test, device, 1.)
            print("err", err)

            pdf_path = "lstm_msp_" + str(i) +".pdf"
            err_path = "lstm_mse_" + str(i) +".pdf"
            plot_multi_traj(X_test, pred, pdf_path)
            plot_multi_traj(err, err, err_path)

            print("multi-step err:", np.linalg.norm(err))

    
    return X_test, test_preds, outputs, torch.tensor(loss_list), torch.tensor(test_loss_list)


###### 1. Data ######

# input to LSTM: (N, L, H_in): N = batch size, L = sequence length, H_in: input feature size
data = load_longer_data()
X_s, y_s = standardize(data)
dim = 10
#X, y = data.reshape(-1, 1), data.reshape(-1, 1)
X, y = split_sequences(X_s, y_s, dim, dim)
print("after split", X.shape, y.shape)
N = X.shape[0]
X_train, X_test, y_train, y_test = train_test_split(N, X, y, 0.7)

# convert to pytorch tensor
X_train = torch.Tensor(X_train) # torch.squeeze(torch.Tensor(X_train))
X_test = torch.Tensor(X_test) # torch.squeeze(torch.Tensor(X_test))
y_train = torch.squeeze(torch.Tensor(y_train))
y_test = torch.squeeze(torch.Tensor(y_test))

print("X_train:", X_train.shape, "\n", "X_test:", X_test.shape, "\n", "y_train:", y_train.shape, "\n", "y_test:", y_test.shape)



##### 2. Compute LE #####

LE_whole = nolds.lyap_r(data)
print("LE", LE_whole)


##### 3. Train #####
warnings.filterwarnings('ignore')

n_epochs = 1000 # 1000 epochs
learning_rate = 1e-3 # 0.001 lr
input_size = 1 # number of features
hidden_size = 2 # number of features in hidden state
num_layers = 1 # number of stacked lstm layers
num_classes = 10 # number of output classes 
loss_fn = torch.nn.MSELoss()

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
optimiser = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)

X_test, test_preds, outputs, loss, test_loss = training_loop(n_epochs=n_epochs, lstm=lstm, optimiser=optimiser, loss_fn=loss_fn, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


###### 4. Save ######
# Save Trained Model
model_path = "../test_result/lstm_model.pt"
torch.save(lstm.state_dict(), model_path)
print("Saved new model!")

# Save whole trajectory
np.savetxt('../test_result/'+"lstm_pred_traj.csv", np.asarray(outputs.detach().cpu()), delimiter=",")

np.savetxt('../test_result/'+"lstm_true_traj.csv", np.asarray(y_train.detach().cpu()), delimiter=",")

np.savetxt('../test_result/'+"lstm_train_loss.csv", np.asarray(loss.detach().cpu()), delimiter=",")

np.savetxt('../test_result/'+"lstm_test_loss.csv", np.asarray(test_loss.detach().cpu()), delimiter=",")



##### 4. Evaluate #####

# split the sequence
X, y = data, data
mm = MinMaxScaler()
ss = StandardScaler()
X = ss.fit_transform(X.reshape(-1, 1))
y = mm.fit_transform(y.reshape(-1, 1))

px, py = split_sequences(X, y, num_classes, num_classes)
# converting to tensors
px, py = torch.Tensor(px), torch.Tensor(py)
# reshaping the dataset
px = torch.reshape(px, (px.shape[0], num_classes, px.shape[2]))

train_predict = lstm(px) # forward pass
data_predict = train_predict.data.numpy() # numpy conversion
dataY_plot = py.data.numpy()

data_predict = mm.inverse_transform(data_predict) # reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
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
pdf_path = "whole_plot_lstm.png"
fig.savefig(pdf_path, format='png', dpi=400)




LE_learned = nolds.lyap_r(preds)
print("LE Learned", LE_learned)

LE_t = nolds.lyap_r(true)
print("LE True", LE_t)

print("Whole Traj Norm Diff:", np.linalg.norm(LE_learned - LE_t))

LE_unseen_p = nolds.lyap_r(preds[train_test_cutoff:])
print("LE Learned unseen", LE_unseen_p)

LE_unseen_t = nolds.lyap_r(true[train_test_cutoff:])
print("LE Learned true", LE_unseen_t)

print("Whole Traj Norm Diff:", np.linalg.norm(LE_unseen_p - LE_unseen_t))
