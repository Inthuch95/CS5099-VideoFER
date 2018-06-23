'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential, load_model
from keras.layers import LSTM, Flatten, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from train_util import load_data_lstm

epochs = 100
batch_size = 8
lstm_unit = 2048
hidden_unit = 512
model_file = '../LSTM_' + str(batch_size) + '_' + str(lstm_unit) + '_' + str(hidden_unit) + '.h5'

def get_network(X_train):
    model = Sequential()
    model.add(LSTM(lstm_unit, return_sequences=False, input_shape=X_train.shape[1:],
                   dropout=0.5))
    model.add(Dense(hidden_unit, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(model, X_train, y_train, X_val, y_val, X_test, y_test): 
    callbacks = [ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=0)]
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=callbacks)
    model = load_model(model_file)
    score = model.evaluate(X_val, y_val)
    print(score)
    return model

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_lstm()
    model = get_network(X_train)
    model = train(model, X_train, y_train, X_val, y_val, X_test, y_test)