'''
Created on Aug 13, 2018

@author: Inthuch Therdchanakul
'''
from LSTMNetwork import LSTMNetwork
from utils import load_vgg_sequence, load_au_sequence, load_game_sequence
import numpy as np
import sys

if __name__ == '__main__':
    feature = 'VGG16'
    data_type = 'Complex'
    n_layer = 1
    lstm_unit = 16
    batch_size = 32
    epochs = 10
    if data_type == 'Game':
        feature = 'VGG16'
        X_train, y_train, X_val, y_val, _, _ = load_vgg_sequence()
        X_train_game, y_train_game, X_val_game, y_val_game, _, _ = load_game_sequence()
        X_train = np.concatenate((X_train, X_train_game))
        y_train = np.concatenate((y_train, y_train_game))
        X_val = np.concatenate((X_val, X_val_game))
        y_val = np.concatenate((y_val, y_val_game))
    else:
        if feature == 'VGG16':
            X_train, y_train, X_val, y_val, _, _ = load_vgg_sequence(data_type=data_type)
        elif feature == 'AU':
            X_train, y_train, X_val, y_val, _, _ = load_au_sequence(data_type=data_type)
        else:
            print("Invalid feature")
            sys.exit()
    lstm_net = LSTMNetwork(n_layer, lstm_unit, X_train.shape[1:], feature, data_type)
#     lstm_net.train(X_train, y_train, X_val, y_val, epochs, batch_size)
#     lstm_net.evaluate(X_val, y_val)
    lstm_net.compare_model(X_val, y_val)