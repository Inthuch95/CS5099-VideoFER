'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from utils import load_data_sequence, plot_confusion_matrix, get_predictions_and_labels, get_network
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

data_type = 'Complex'
# data_type = 'Basic'
if data_type == 'Basic':
    DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
elif data_type == 'Complex':
    DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
else:
    print("Invalid data type")
    sys.exit()
EMOTIONS = DATA['EMOTIONS']
batch_size = 256
epochs = 100
n_layers = 3
lstm_unit = 512
current_time = time.strftime("%Y%m%d-%H%M%S")
model_dir = 'LSTM_' + str(n_layers) + '_' + str(lstm_unit) + '_' + current_time + '/'
filename = 'LSTM_' + str(n_layers) + '_' + str(lstm_unit) + '_' + current_time + '.h5'
base_dir = '../LSTM/' + data_type + '/'
model_file = base_dir + model_dir + filename

def train(model, X_train, y_train, X_val, y_val): 
    # compile and train the model
    log_dir = base_dir + model_dir + 'log/'
    os.mkdir(log_dir)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=0),
                 TensorBoard(log_dir=log_dir, write_graph=True)]
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=callbacks)
    return model

def evaluate(X_val, y_val):
    # evaluate the model with validation set
    model = load_model(model_file)
    scores = model.evaluate(X_val, y_val)
    print('val_loss: {}, val_acc: {}'.format(scores[0], scores[1]))
    
    y_true, y_pred = get_predictions_and_labels(model, X_val, y_val)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(cm_percent, index=EMOTIONS, columns=EMOTIONS)
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    df.to_csv(base_dir+model_dir+'confusion_matrix_validation.csv', float_format='%.4f')
    
    if data_type == 'Basic':
        # plot percentage confusion matrix
        fig1, ax1 = plt.subplots()
        plot_confusion_matrix(cm_percent, class_names=EMOTIONS)
        plt.savefig(base_dir + model_dir + 'cm_percent_val.png', format='png')
        # plot normal confusion matrix
        fig2, ax2 = plt.subplots()
        plot_confusion_matrix(cm, float_display='.0f', class_names=EMOTIONS)
        plt.savefig(base_dir + model_dir + 'cm_val.png', format='png')
        
        plt.show()
    
def compare_model(X_val, y_val):
    folder_list = [model_dir for model_dir in os.listdir(base_dir) if 'LSTM' in model_dir]
    for folder in folder_list:
        filename = folder + '.h5'
        path = os.path.join(base_dir, folder, filename)
        model = load_model(path)
        scores = model.evaluate(X_val, y_val)
        print('model: {}, val_loss: {}, val_acc: {}'.format(folder, scores[0], scores[1]))

if __name__ == '__main__':
    if not os.path.exists(base_dir + model_dir):
        os.mkdir(base_dir + model_dir)
    X_train, y_train, X_val, y_val, _, _ = load_data_sequence(data_type=data_type)
    model = get_network(n_layers, X_train.shape[1:], lstm_unit, len(EMOTIONS))
    model = train(model, X_train, y_train, X_val, y_val)
    evaluate(X_val, y_val)
    compare_model(X_val, y_val)
    