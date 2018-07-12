'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import load_data_sequence, plot_confusion_matrix, get_predictions_and_labels
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import sys

data_type = 'Basic'
# data_type = 'Complex'
if data_type == 'Basic':
    DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
elif data_type == 'Complex':
    DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
else:
    print("Invalid data type")
    sys.exit()
EMOTIONS = DATA['EMOTIONS']
base_dir = '../best model/' + data_type + '/'
model_file =  base_dir + 'LSTM_best/LSTM_best.h5'
# model_file = '../best model/LSTM_best/LSTM_best.h5'

if __name__ == '__main__':
    _, _, _, _, X_test, y_test = load_data_sequence(data_type=data_type)
    # evaluate the model with test set
    model = load_model(model_file)
    print(model.summary())
    scores = model.evaluate(X_test, y_test)
    print('test_loss: %.4f, test_acc: %.4f' % (scores[0], scores[1]))
    
    y_true, y_pred = get_predictions_and_labels(model, X_test, y_test)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot percentage confusion matrix
    if data_type == 'Basic':
        fig1, ax1 = plt.subplots()
        plot_confusion_matrix(cm_percent, title='LSTM', class_names=EMOTIONS)
        plt.savefig(base_dir+'LSTM_best/cm_percent_test.png')    
        plt.show()
        # plot normal confusion matrix
        fig2, ax2 = plt.subplots()
        plot_confusion_matrix(cm, title='LSTM', float_display='.0f', class_names=EMOTIONS)
        plt.savefig(base_dir+'LSTM_best/cm_test.png')
        
    df = pd.DataFrame(cm_percent, index=EMOTIONS, columns=EMOTIONS)
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    df.to_csv(base_dir+'LSTM_best/confusion_matrix_test.csv', float_format='%.4f')