'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from train_util import load_sequence_lstm
import os

epochs = 50
batch_size = 32
lstm_unit = 512
hidden_unit = 1024
feature = 'vgg16'
sub_dir = 'LSTM_' + str(lstm_unit) + '_' + str(hidden_unit) + '/'
filename = 'LSTM_' + str(lstm_unit) + '_' + str(hidden_unit) + '.h5'
model_file = '../LSTM/' + sub_dir + filename

def get_network(X_train):
    model = Sequential()
    model.add(LSTM(lstm_unit, return_sequences=False, input_shape=X_train.shape[1:],
                   dropout=0.2))
    if hidden_unit > 0:
        model.add(Dense(hidden_unit, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_vgg(model, X_train, y_train, X_val, y_val, X_test, y_test): 
    callbacks = [ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=0),
                 TensorBoard(log_dir='../Log', histogram_freq=0, write_graph=True, write_images=True)]
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=callbacks)
    model = load_model(model_file)
    score = model.evaluate(X_val, y_val)
    print(score)
    return model

def train_sequence(model, X_train, y_train, X_test, y_test):
    callbacks = [ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=0),
                 TensorBoard(log_dir='../Log', histogram_freq=0, write_graph=True, write_images=True)]
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=callbacks)
    model = load_model(model_file)
    score = model.evaluate(X_test, y_test)
    print(score)
    return model

if __name__ == '__main__':
    child_dir = 'LSTM_' + str(lstm_unit) + '_' + str(hidden_unit)
    if child_dir not in os.listdir('../LSTM/'):
        os.mkdir('../LSTM/' + sub_dir)
#     X_train, y_train, X_val, y_val, X_test, y_test = load_vgg_data_lstm()
    X_train, y_train, X_test, y_test = load_sequence_lstm(feature)
    model = get_network(X_train)
    model = train_sequence(model, X_train, y_train, X_test, y_test)
#     model = load_model(model_file)
#     score = model.evaluate(X_test, y_test)
    