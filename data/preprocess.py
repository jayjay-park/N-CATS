import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import glob

# 1. preprocessing
# 9/17/2014 - 11/23/2023 | Daily BTC-USD
# From https://finance.yahoo.com/quote/BTC-USD/history?period1=1410912000&period2=1642118400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true&guccounter=1&guce_referrer=aHR0cHM6Ly9jaGFybGllb25laWxsLm1lZGl1bS5jb20vcHJlZGljdGluZy10aGUtcHJpY2Utb2YtYml0Y29pbi13aXRoLW11bHRpdmFyaWF0ZS1weXRvcmNoLWxzdG1zLTY5NWJjMjk0MTMw&guce_referrer_sig=AQAAALY2qpXJUnFlLHioUfmcy6E-DzEvcPnCN8ps-4W8rYvnTKnIpZN6QDIjalFDx5HRdEfL7Fx4Cp5KRvf6nLr4gNib7jmeVsTW7WlVo1lUOMFBGcJcfS2gsZIIQm0W1NAJaS-Kp36djGnjCdoqHACDEsHFWC4I80bkxsX-KUT3WN7Q

def load_data():
    # Load dataset
    df = pd.read_csv('../data/BTC-USD.csv', index_col = 'Date', parse_dates=True)
    df.drop(columns=['Adj Close'], inplace=True)
    data = df.Close.values
    print("data shape:", data.shape)

    # Understand data
    # plot(data)
    # xlabel("Time")
    # ylabel("Price (USD)")
    # title("Bitcoin price over time")
    # savefig("initial_plot.png", dpi=250)

    return data

def load_longer_data():
    # https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd/

    # Specify the columns you want to read
    col = ['close']
    dtype = {'close': 'float32'}

    # df_2017 = pd.read_csv('../data/BTC-2017min.csv', usecols=col)
    # df_2018 = pd.read_csv('../data/BTC-2018min.csv', usecols=col)
    # df_2019 = pd.read_csv('../data/BTC-2019min.csv', usecols=col)
    # df_2020 = pd.read_csv('../data/BTC-2020min.csv', usecols=col, dtype=dtype)
    df_2021 = pd.read_csv('../data/BTC-2021min.csv', usecols=col, dtype=dtype)

    list_df = [df_2021]
    df = pd.concat(list_df, axis=0, ignore_index=True)

    data = df.close.values[600000:]
    print("data shape:", data.shape)
    return data



def split_sequences(input_seq, output_seq, n_steps_in, n_steps_out):
    ''' Create function that feed in 50 samples up to current day, and predict the next 50 time step values '''

    X, y = list(), list() # instantiate X and y
    for i in range(len(input_seq)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1

        # check if we are beyond the dataset
        if out_end_ix > len(input_seq): 
            print("out_end_ix", out_end_ix)
            break
        # gather input and output of the pattern
        seq_x, seq_y = input_seq[i:end_ix], output_seq[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


def standardize(data):
    # Standardising features: remove the mean and scaling to unit variance
    # Standardisation helps the deep learning model to learn by ensuring that parameters can exist in the same multi-dimensional space
    # For y, we will scale and translate each feature individually to between 0 and 1. This transformation is often used as an alternative to zero mean, unit variance scaling.

    X, y = data, data
    mm = MinMaxScaler()
    ss = StandardScaler()
    X_trans = mm.fit_transform(X.reshape(-1, 1)) # ss
    y_trans = mm.fit_transform(y.reshape(-1, 1))
    return X_trans, y_trans


def train_test_split(total_samples, X_ss, y_mm, partition):
    # We want to predict the data a several months into the future. Training data size of 95% with 5% predict
    # This gives us a training set size of XXX days

    train_test_cutoff = round(partition * total_samples)
    # print("index", train_test_cutoff)
    X_train = X_ss[:train_test_cutoff]
    X_test = X_ss[train_test_cutoff:]
    y_train = y_mm[:train_test_cutoff]
    y_test = y_mm[train_test_cutoff:] 
    # print("Training Shape:", X_train.shape, y_train.shape)
    # print("Testing Shape:", X_test.shape, y_test.shape)
    # index 3020
    # Training Shape: (3020, 100, 4) (3020, 50)
    # Testing Shape: (187, 100, 4) (187, 50)

    return X_train, X_test, y_train, y_test


def train_test_split_1d(total_samples, X_ss, y_mm, partition):
    # We want to predict the data a several months into the future. Training data size of 95% with 5% predict
    # This gives us a training set size of XXX days

    train_test_cutoff = round(partition * total_samples)
    # print("index", train_test_cutoff)
    X_train = X_ss[:train_test_cutoff-1]
    y_train = y_mm[1:train_test_cutoff]
    X_test = X_ss[train_test_cutoff:-1]
    y_test = y_mm[train_test_cutoff+1:] 
    # print("Training Shape:", X_train.shape, y_train.shape)
    # print("Testing Shape:", X_test.shape, y_test.shape)
    # index 3020
    # Training Shape: (3020, 100, 4) (3020, 50)
    # Testing Shape: (187, 100, 4) (187, 50)

    return X_train, X_test, y_train, y_test
