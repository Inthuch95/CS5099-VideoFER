'''
Created on Aug 11, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from shutil import copyfile
from keras.preprocessing import image
import numpy as np
import os
import pickle

DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
EMOTIONS = DATA['EMOTIONS']
IMG_WIDTH, IMG_HEIGHT = 100,100
SEQ_LENGTH = 2
base_dir = '../best model/Basic/VGG16/'
model_file =  base_dir + 'LSTM_best.h5'

def extract_features(model, image_path):
    # load and preprocess the frame
    img = image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the prediction.
    features = model.predict(x)
    features = features[0]
    return features

if __name__ == '__main__':
    model = load_model(model_file)
    vgg_model = VGG16(include_top=False, weights='imagenet') 
    for meeting in os.listdir('../meeting_data/'):
        for person in os.listdir('../meeting_data/' + meeting + '/'):
            frame_path = '../meeting_data/' + meeting + '/' + person + '/'
            frames = [f for f in os.listdir(frame_path) if '.png' in f or '.jpg' in f]
            if len(frames) >= SEQ_LENGTH:
                sequence = []
                filename = []      
                for frame in frames:
                    img_file = frame_path + frame
                    features = extract_features(vgg_model, img_file)
                    filename.append(frame)
                    sequence.append(features)
                    if len(sequence) == SEQ_LENGTH:
                        X = np.array(sequence)
                        X = X.reshape(1, X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
                        pred = model.predict(X)
                        pred_idx = np.argmax(pred)
                        for f in filename:
                            src = frame_path + f
                            dst = frame_path + EMOTIONS[pred_idx] + '/' + f
                            copyfile(src, dst)
                        filename = []
                        sequence  = []