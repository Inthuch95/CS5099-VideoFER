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

DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
EMOTIONS = DATA['EMOTIONS']
data_type='Complex'
base_dir = '../best model/Complex/'
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
    # plot normal confusion matrix
#     fig1, ax1 = plt.subplots()
#     plot_confusion_matrix(cm, title='LSTM', float_display='.0f', class_names=emotions)
    # plot percentage confusion matrix
    if data_type == 'Basic':
        fig, ax = plt.subplots()
        plot_confusion_matrix(cm_percent, title='LSTM', class_names=EMOTIONS)    
        plt.show()
    df = pd.DataFrame(cm_percent, index=EMOTIONS, columns=EMOTIONS)
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    df.to_csv(base_dir+'LSTM_best/confusion_matrix_test.csv', float_format='%.4f')