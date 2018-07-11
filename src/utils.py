'''
Created on Jun 23, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
        
def plot_confusion_matrix(cm, title='Confusion matrix', float_display='.4f', cmap=plt.cm.Greens, class_names=None):
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

def load_data_sequence(data_type='Basic'):
    # load data
    if data_type == 'Basic':
        base_dir = '../prepared_data/Basic/'
    else:
        base_dir = '../prepared_data/Complex/'
    X = np.load(base_dir+'sequence/X_vgg16.npy')
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4])
    y = np.load(base_dir+'sequence/y_vgg16.npy')
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, test_size=0.2)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_data_single(data_type='Basic'):
    # load data
    if data_type == 'Basic':
        base_dir = '../prepared_data/Basic/'
    else:
        base_dir = '../prepared_data/Complex/'
    X = np.load(base_dir+'single/X_vgg16.npy')
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]*X.shape[4])
    y = np.load(base_dir+'single/y_vgg16.npy')
    X_train, y_train, X_test, y_test = split_dataset(X, y, test_size=0.2, val_split=False)
    return X_train, y_train, X_test, y_test

def split_dataset(X, y, test_size=0.2, val_split=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    if val_split:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

def save_deleted_frames(data_type='Basic'):
    if data_type == 'Basic':
        data_path = '../prepared_data/Basic/deleted/'
        deleted_frames = [f for f in os.listdir(data_path)]
        np.save('../basic_deleted_frames.npy', deleted_frames)
    elif data_type == 'Complex':
        data_path = '../prepared_data/Complex/deleted/'
        deleted_frames = [f for f in os.listdir(data_path)]
        np.save('../complex_deleted_frames.npy', deleted_frames)
    
def replace_whitespace(parent):
    # remove whitespace from the video dataset
    for path, folders, files in os.walk(parent):
        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.replace(' ', '_')))
        for i in range(len(folders)):
            new_name = folders[i].replace(' ', '_')
            os.rename(os.path.join(path, folders[i]), os.path.join(path, new_name))
            folders[i] = new_name
            
def save_var_data(data_type='Basic'):
    if data_type == 'Complex':
        data_dict = {'EMOTIONS': [f for f in os.listdir('../EUDataComplexVideo/') if os.path.isdir('../EUDataComplexVideo/'+f)], 
                     'DELETED_FRAMES': np.load('../complex_deleted_frames.npy'),
                     'VIDEO_PATH': '../EUDataComplexVideo/',
                     'EXTRACT_PATH': '../video_frames/Complex/',
                     'DATA_PATH': '../prepared_data/Complex/data/',
                     'SEQUENCE_PATH': '../prepared_data/Complex/sequence/',
                     'SINGLE_PATH': '../prepared_data/Complex/single/',
                     'DELETED_PATH': '../prepared_data/Complex/deleted/'}
        with open('../complex_emotions_data.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    elif data_type == 'Basic':
        data_dict = {'EMOTIONS': [f for f in os.listdir('../EUDataBasicVideo/') if os.path.isdir('../EUDataBasicVideo/'+f)], 
                   'DELETED_FRAMES': np.load('../basic_deleted_frames.npy'),
                   'VIDEO_PATH': '../EUDataBasicVideo/',
                   'EXTRACT_PATH': '../video_frames/Basic/',
                   'DATA_PATH': '../prepared_data/Basic/data/',
                   'SEQUENCE_PATH': '../prepared_data/Basic/sequence/',
                   'SINGLE_PATH': '../prepared_data/Basic/single/',
                   'DELETED_PATH': '../prepared_data/Basic/deleted/'}
        with open('../basic_emotions_data.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        print('Invalid parameter')

def get_network(n_layers, input_shape, lstm_unit, nb_class):
    model = Sequential()
    if n_layers > 1:
        model.add(LSTM(lstm_unit, return_sequences=True, input_shape=input_shape,
                   dropout=0.2))
        layer_count = 1
        while layer_count < n_layers:
            if layer_count == n_layers-1:
                model.add(LSTM(lstm_unit, return_sequences=False, dropout=0.2))
            else:
                model.add(LSTM(lstm_unit, return_sequences=True, dropout=0.2))
            layer_count += 1
    else:
        model.add(LSTM(lstm_unit, return_sequences=False, input_shape=input_shape,
                   dropout=0.2))
    model.add(Dense(nb_class, activation='softmax'))
    return model

if __name__ == '__main__':
    for i in range(1, 1000):
        if 18092 % i == 0:
            print(i)