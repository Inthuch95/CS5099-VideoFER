'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from utils import load_data, plot_confusion_matrix, get_predictions_and_labels, get_network
import time
import numpy as np
import matplotlib.pyplot as plt
import os

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
epochs = 100
batch_size = 32
lstm_unit = 512
current_time = time.strftime("%Y%m%d-%H%M%S")
model_dir = 'LSTM_' + str(lstm_unit) + '_' + current_time + '/'
filename = 'LSTM_' + str(lstm_unit) + '_' + current_time + '.h5'
model_file = '../LSTM/' + model_dir + filename

def train(model, X_train, y_train, X_val, y_val): 
    # compile and train the model
    log_dir = '../LSTM/' + model_dir + 'log/'
    os.mkdir(log_dir)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=0),
                 TensorBoard(log_dir=log_dir, histogram_freq=10, write_graph=True)]
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
    # plot percentage confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm_percent, class_names=emotions)
    plt.savefig('../LSTM/' + model_dir + 'cm_percent_val.png', format='png')
    # plot normal confusion matrix
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(cm, float_display='.0f', class_names=emotions)
    plt.savefig('../LSTM/' + model_dir + 'cm_val.png', format='png')
    
    plt.show()
    
def compare_model(X_val, y_val):
    folder_list = [model_dir for model_dir in os.listdir('../LSTM/') if 'LSTM' in model_dir]
    for folder in folder_list:
        filename = folder + '.h5'
        path = os.path.join('../LSTM', folder, filename)
        model = load_model(path)
        scores = model.evaluate(X_val, y_val)
        print('model: {}, val_loss: {}, val_acc: {}'.format(folder, scores[0], scores[1]))

if __name__ == '__main__':
#     if not os.path.exists('../LSTM/'+ model_dir):
#         os.mkdir('../LSTM/' + model_dir)
    X_train, y_train, X_val, y_val, _, _ = load_data()
#     model = get_network(X_train, lstm_unit)
#     model = train(model, X_train, y_train, X_val, y_val)
#     evaluate(X_val, y_val)
    compare_model(X_val, y_val)
    