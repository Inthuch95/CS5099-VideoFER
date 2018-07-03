'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import load_data, plot_confusion_matrix, get_predictions_and_labels
import matplotlib.pyplot as plt
import numpy as np

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
model_file = '../best model/LSTM_best/LSTM_best.hdf5'

if __name__ == '__main__':
    _, _, _, _, X_test, y_test = load_data()
    # evaluate the model with test set
    model = load_model(model_file)
    print(model.summary())
    scores = model.evaluate(X_test, y_test)
    print('test_loss: %.4f, test_acc: %.4f' % (scores[0], scores[1]))
    
    y_true, y_pred = get_predictions_and_labels(model, X_test, y_test)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot percentage confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm, float_display='.0f', class_names=emotions)
    # plot normal confusion matrix
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(cm_percent, class_names=emotions)    
    plt.show()
