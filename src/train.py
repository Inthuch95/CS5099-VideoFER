'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from utils import load_sequence_lstm, plot_confusion_matrix, get_predictions_and_labels
import time
import numpy as np
import matplotlib.pyplot as plt
import os

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
epochs = 100
batch_size = 32
lstm_unit = 256
current_time = time.strftime("%Y%m%d-%H%M%S")
model_dir = 'LSTM_' + str(lstm_unit) + '_' + current_time + '/'
filename = 'LSTM_' + str(lstm_unit) + '_' + current_time + '.hdf5'
model_file = '../LSTM/' + model_dir + filename

def get_network(X_train):
    model = Sequential()
    model.add(LSTM(lstm_unit, return_sequences=False, input_shape=X_train.shape[1:],
                   dropout=0.2))
    model.add(Dense(7, activation='softmax'))
    return model

def train(model, X_train, y_train, X_val, y_val): 
    log_dir = '../LSTM/' + model_dir + 'log/'
    os.mkdir(log_dir)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=0),
                 TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)]
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=callbacks)
    return model

def evaluate(X_val, y_val):
    model = load_model(model_file)
    scores = model.evaluate(X_val, y_val)
    print('val_loss: {}, vall_acc: {}'.format(scores[0], scores[1]))
    y_true, y_pred = get_predictions_and_labels(model, X_val, y_val)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, class_names=emotions)
    plt.show()

if __name__ == '__main__':
    if model_dir not in os.listdir('../LSTM/'):
        os.mkdir('../LSTM/' + model_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = load_sequence_lstm()
    model = get_network(X_train)
    model = train(model, X_train, y_train, X_val, y_val)
    evaluate(X_val, y_val)
    