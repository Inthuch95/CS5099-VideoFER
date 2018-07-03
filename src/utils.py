'''
Created on Jun 23, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential
from keras.layers import LSTM, Dense
import itertools
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, title='Confusion matrix', float_display='.3f', cmap=plt.cm.Reds, class_names=None):
    # create confusion matrix plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels(class_names)
    plt.yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], float_display),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
def get_predictions_and_labels(model, X, y):
    predictions = model.predict(X)
    y_true = []
    y_pred = []
    for i in range(len(y)):
        label = list(y[i]).index(1)
        pred = list(predictions[i])
        max_value = max(pred)
        max_index = pred.index(max_value)
        p = max_index
        y_true.append(label)
        y_pred.append(p)    
    return y_true, y_pred    

def load_data():
    # load train data
    X_train = np.load(open('../prepared_data/sequence/X_train_vgg16.npy', 'rb'))
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2]*X_train.shape[3]*X_train.shape[4])
    y_train = np.load('../prepared_data/sequence/y_train_vgg16.npy')
    # load train data
    X_val = np.load(open('../prepared_data/sequence/X_val_vgg16.npy', 'rb'))
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2]*X_val.shape[3]*X_val.shape[4])
    y_val = np.load('../prepared_data/sequence/y_val_vgg16.npy')      
    #load test data
    X_test = np.load(open('../prepared_data/sequence/X_test_vgg16.npy', 'rb'))
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2]*X_test.shape[3]*X_test.shape[4])
    y_test = np.load('../prepared_data/sequence/y_test_vgg16.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_data_svm():
    # load train data
    X_train = np.load('../prepared_data/single/X_train_vgg16.npy')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
    y_train = np.load('../prepared_data/single/y_train_vgg16.npy')
    #load test data
    X_test = np.load('../prepared_data/single/X_test_vgg16.npy')
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
    y_test = np.load('../prepared_data/single/y_test_vgg16.npy')
    return X_train, y_train, X_test, y_test

def get_network(X_train, lstm_unit):
    model = Sequential()
    model.add(LSTM(lstm_unit, return_sequences=False, input_shape=X_train.shape[1:],
                   dropout=0.2))
    model.add(Dense(7, activation='softmax'))
    return model

if __name__ == '__main__':
    pass