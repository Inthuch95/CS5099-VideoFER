'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dropout, Dense
import numpy as np
import os

selected_model = "lstm"
epochs = 50
batch_size = 1

def load_data():
    # load training data
    path = '../dataFromPython3/train/'
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
    emotionLbls = [[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0],[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]]
    train_data = np.load(open('../bottleneck_features_train.npy', 'rb'))
    labelbyemotion = []
    for emotion in emotions:
        for file in os.listdir(path+emotion):
            if 'png' in file:
                labelbyemotion.append(emotionLbls[emotions.index(emotion)])
    y_train = np.array(labelbyemotion)
    
    #load validation data
    validation_data = np.load(open('../bottleneck_features_validation.npy', 'rb'))
    path = '../dataFromPython3/validation/'
    labelbyemotionVal = []
    for emotion in emotions:
        for file in os.listdir(path + emotion):
            if 'png' in file:
                labelbyemotionVal.append(emotionLbls[emotions.index(emotion)])
    y_test = np.array(labelbyemotionVal)
    
    X_train = np.zeros((144, 3, 3, 512), dtype=np.uint8)
    X_train = np.reshape(train_data, (144, 3, 3*512))
    X_test = np.zeros((38, 3, 3, 512), dtype=np.uint8)
    X_test = np.reshape(validation_data, (38, 3, 3*512))
    return X_train, y_train, X_test, y_test

def get_network(X_train):
    model = Sequential()
    # model.add(Flatten(input_shape=train_data.shape[1:]))
    if selected_model == "lstm":
        model.add(LSTM(2048, return_sequences=False, input_shape=X_train.shape[1:],
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
    elif selected_model == "mlp":
        model.add(Flatten(input_shape=X_train.shape[1:]))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(6, activation='softmax'))
    
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(model, X_train, y_train, X_test, y_test): 
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test)
    print(score)
    return model

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open( "../PretrainLSTM.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../PreTraineLSTMWeight.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    model = get_network(X_train)
    model = train(model, X_train, y_train, X_test, y_test)
#     save_model(model)