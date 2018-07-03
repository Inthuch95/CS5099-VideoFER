'''
Created on Jun 27, 2018

@author: Inthuch Therdchanakul
'''
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import random
import os

img_width, img_height = 100,100
seq_length = 2
data_path = '../prepared_data/Emotions/'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotionLbls = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0], [0,0,1,0,0,0,0],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

# get sequence of features for RNN
def extract_feature_sequence(model):
    X, y = [], []
    for emotion in emotions:
        video_list = [f for f in os.listdir(data_path + emotion)]
        for video in video_list:
            video_path = data_path + emotion + '/' + video
            frames = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
            if len(frames) >= seq_length:
                sequence = []
                for frame in frames:
                    frame = video_path + '/' + frame
                    features = extract_features(model, frame)
                    sequence.append(features)
                    if len(sequence) == seq_length:
                        X.append(sequence)
                        y.append(emotionLbls[emotions.index(emotion)])
                        sequence = []
        print('{} sequences extracted'.format(emotion))
    print('Saving sequence')
    save_feature_sequence(X, y)

def save_feature_sequence(X, y, test_split=0.1):
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y, test_split=test_split)
            
    print(X_train.shape)
    np.save('../prepared_data/sequence/X_train_vgg16.npy', X_train)
    np.save('../prepared_data/sequence/y_train_vgg16.npy', y_train)
    
    print(X_val.shape)
    np.save('../prepared_data/sequence/X_val_vgg16.npy', X_val)
    np.save('../prepared_data/sequence/y_val_vgg16.npy', y_val)
    
    print(X_test.shape)
    np.save('../prepared_data/sequence/X_test_vgg16.npy', X_test)
    np.save('../prepared_data/sequence/y_test_vgg16.npy', y_test)

def extract_features(model, image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the prediction.
    features = model.predict(x)
    features = features[0]
    return features

def train_test_split(X, y, test_split=0.1):
    data = list(zip(X, y))
    random.seed(42)
    random.shuffle(data)
    X, y = zip(*data)
    split_2 = 1.0 - test_split
    split_1 = split_2 - test_split 
    split_1 = int(split_1 * len(X)) 
    split_2 = int(split_2 * len(X))
    
    X_train = X[:split_1]
    y_train = y[:split_1]
    X_val = X[split_1:split_2]
    y_val = y[split_1:split_2]
    X_test = X[split_2:]
    y_test = y[split_2:]
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    model = VGG16(include_top=False, weights='imagenet')
    extract_feature_sequence(model)
    print('Sequences saved')