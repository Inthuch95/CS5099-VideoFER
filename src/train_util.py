'''
Created on Jun 23, 2018

@author: Inthuch Therdchanakul
'''
import numpy as np

def load_vgg_data_lstm():
    # load train_lstm data
    X_train = np.load(open('../dataset/bottleneck_features_train.npy', 'rb'))
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2], X_train.shape[3])
    y_train = np.load('../dataset/y_train.npy')    
    #load validation data
    X_val = np.load(open('../dataset/bottleneck_features_validation.npy', 'rb'))
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2], X_val.shape[3])
    y_val = np.load('../dataset/y_val.npy')
    #load test data
    X_test = np.load(open('../dataset/bottleneck_features_test.npy', 'rb'))
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2], X_test.shape[3])
    y_test = np.load('../dataset/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_sequence_lstm(feature='vgg16'):
    if feature == 'inception':
        # load train_lstm data
        X_train = np.load(open('../dataset2/X_train_inception.npy', 'rb'))
        y_train = np.load('../dataset2/y_train_inception.npy')    
        #load test data
        X_test = np.load(open('../dataset2/X_test_inception.npy', 'rb'))
        y_test = np.load('../dataset2/y_test_inception.npy')
    else:
        # load train_lstm data
        X_train = np.load(open('../dataset2/X_train_vgg16.npy', 'rb'))
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2]*X_train.shape[3]*X_train.shape[4])
        y_train = np.load('../dataset2/y_train_vgg16.npy')    
        #load test data
        X_test = np.load(open('../dataset2/X_test_vgg16.npy', 'rb'))
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2]*X_test.shape[3]*X_test.shape[4])
        y_test = np.load('../dataset2/y_test_vgg16.npy')
    return X_train, y_train, X_test, y_test

def load_data_svm():
    # load train_lstm data
    X_train = np.load(open('../dataset/bottleneck_features_train.npy', 'rb'))
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
    y_train = np.load('../dataset/y_train_text.npy')
    #load test data
    X_test = np.load(open('../dataset/bottleneck_features_test.npy', 'rb'))
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
    y_test = np.load('../dataset/y_test_text.npy')
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    pass