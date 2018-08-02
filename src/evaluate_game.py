'''
Created on Aug 2, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, get_predictions_and_labels, load_game_data, load_data_sequence
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import sys

DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
EMOTIONS = DATA['EMOTIONS']
save_dir = '../best model/Game/offline/'
model_file =  save_dir + 'LSTM_best.h5'

if __name__ == '__main__':
    eval_mode = 'game'
    if eval_mode == 'game':
        _, _, X_val, y_val, X_test, y_test = load_game_data()
    elif eval_mode == 'all':
        _, _, X_val, y_val, X_test, y_test = load_data_sequence()
        _, _, X_val_game, y_val_game, X_test_game, y_test_game = load_game_data()
        X_val = np.concatenate((X_val, X_val_game))
        y_val = np.concatenate((y_val, y_val_game))
        X_test = np.concatenate((X_test, X_test_game))
        y_test = np.concatenate((y_test, y_test_game))
    elif eval_mode == 'eess':
        _, _, X_val, y_val, X_test, y_test = load_data_sequence()
    else:
        print('Invalid evaluation option')
        sys.exit()
    print(X_test.shape)
    print(y_test.shape)
    # evaluate the model with test set
    model = load_model(model_file)
    print(model.summary())
    test_scores = model.evaluate(X_test, y_test)
    val_scores = model.evaluate(X_val, y_val)
    print('val_loss: %.4f, val_acc: %.4f' % (val_scores[0], val_scores[1]))
    print('test_loss: %.4f, test_acc: %.4f' % (test_scores[0], test_scores[1]))
    
    y_true, y_pred = get_predictions_and_labels(model, X_test, y_test)
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(EMOTIONS))])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot percentage confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm_percent, title='LSTM', class_names=EMOTIONS)
    plt.savefig(save_dir+'cm_percent_test.png')    
    plt.show()
    # plot normal confusion matrix
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(cm, title='LSTM', float_display='.0f', class_names=EMOTIONS)
    plt.savefig(save_dir+'cm_test.png')
        
    df = pd.DataFrame(cm_percent, index=EMOTIONS, columns=EMOTIONS)
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    df.to_csv(save_dir+'confusion_matrix_test.csv', float_format='%.4f')