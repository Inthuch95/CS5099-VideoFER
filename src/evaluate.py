'''
Created on Aug 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import load_vgg_sequence, load_au_sequence, load_game_sequence, plot_confusion_matrix, \
    get_predictions_and_labels, load_var
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def load_test_data(data_type, feature, eval_mode='game'):
    if data_type == 'Game':
        feature = 'VGG16'
        if eval_mode == 'game':
            _, _, X_val, y_val, X_test, y_test = load_game_sequence()
        elif eval_mode == 'all':
            _, _, X_val, y_val, X_test, y_test = load_vgg_sequence()
            _, _, X_val_game, y_val_game, X_test_game, y_test_game = load_game_sequence()
            X_val = np.concatenate((X_val, X_val_game))
            y_val = np.concatenate((y_val, y_val_game))
            X_test = np.concatenate((X_test, X_test_game))
            y_test = np.concatenate((y_test, y_test_game))
        elif eval_mode == 'eess':
            _, _, X_val, y_val, X_test, y_test = load_vgg_sequence()
        else:
            print('Invalid evaluation option')
            sys.exit()
    else:
        if feature == 'VGG16':
            _, _, X_val, y_val, X_test, y_test = load_vgg_sequence(data_type=data_type)
        elif feature == 'AU':
            _, _, X_val, y_val, X_test, y_test = load_au_sequence(data_type=data_type)
        else:
            print("Invalid feature")
            sys.exit()
    return X_val, y_val, X_test, y_test

def display_scores(model, X_val, y_val, X_test, y_test):
    val_scores = model.evaluate(X_val, y_val)
    test_scores = model.evaluate(X_test, y_test)
    print('evaluated with {} test samples'.format(X_test.shape[0]))
    print('val_loss: %.4f, val_acc: %.4f' % (val_scores[0], val_scores[1]))
    print('test_loss: %.4f, test_acc: %.4f' % (test_scores[0], test_scores[1]))
    
def save_results(model, X_test, y_test, EMOTIONS, base_dir):
    y_true, y_pred = get_predictions_and_labels(model, X_test, y_test)
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(EMOTIONS))])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot percentage confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm_percent, title='LSTM', class_names=EMOTIONS)
    plt.savefig(base_dir+'cm_percent_test.png')    
    plt.show()
    # plot normal confusion matrix
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(cm, title='LSTM', float_display='.0f', class_names=EMOTIONS)
    plt.savefig(base_dir+'cm_test.png')
        
    df = pd.DataFrame(cm_percent, index=EMOTIONS, columns=EMOTIONS)
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    df.to_csv(base_dir+'confusion_matrix_test.csv', float_format='%.4f')

if __name__ == '__main__':
    feature = 'VGG16'
    data_type = 'Game'
    eval_mode = 'game'
    DATA = load_var(data_type)
    EMOTIONS = DATA['EMOTIONS']
    base_dir = '../best model/' + data_type + '/' + feature + '/'
    model_file =  base_dir + 'LSTM_best.h5'
    
    # evaluate the model with test set
    X_val, y_val, X_test, y_test = load_test_data(data_type, feature, eval_mode=eval_mode)
    model = load_model(model_file)
    print(model.summary())
    display_scores(model, X_val, y_val, X_test, y_test)
    save_results(model, X_test, y_test, EMOTIONS, base_dir)