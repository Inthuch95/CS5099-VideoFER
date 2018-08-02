'''
Created on Jun 27, 2018

@author: Inthuch Therdchanakul
'''
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
import pickle
import os
import sys

data_type = 'Basic'
# data_type = 'Complex'
if data_type == 'Basic':
    DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
elif data_type == 'Complex':
    DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
else:
    print("Invalid data type")
    sys.exit()
IMG_WIDTH, IMG_HEIGHT = 100,100
SEQ_LENGTH = 2
OVERLAP_IDX = int(0.9 * SEQ_LENGTH)
DATA_PATH = DATA['DATA_PATH']
EMOTIONS = DATA['EMOTIONS']
DELETED_FRAMES = DATA['DELETED_FRAMES']
SEQUENCE_PATH = DATA['SEQUENCE_PATH']
SINGLE_PATH = DATA['SINGLE_PATH']

# get sequence of features for RNN
def extract_feature_sequence(model):
    X, y = [], []
    for emotion in EMOTIONS:
        video_list = [f for f in os.listdir(DATA_PATH + emotion)]
        for video in video_list:
            video_path = DATA_PATH + emotion + '/' + video
            frames = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
            if len(frames) >= SEQ_LENGTH:
                X, y = process_frames(frames, video_path, emotion, X, y)
        print('{} sequences extracted'.format(emotion))
    # use onehot encoding for LSTM
    if SEQ_LENGTH > 1:
        y = to_categorical(y, num_classes=len(EMOTIONS))
    # save to binary files
    print('Saving sequence')
    if SEQ_LENGTH == 1:
        np.save(SINGLE_PATH+'X_vgg16.npy', X)
        np.save(SINGLE_PATH+'y_vgg16.npy', y)
    else:
        np.save(SEQUENCE_PATH+'X_vgg16.npy', X)
        np.save(SEQUENCE_PATH+'y_vgg16.npy', y)

def process_frames(frames, video_path, emotion, X, y):
    sequence = []      
    for frame in frames:
        # exclude neutral frames 
        if frame not in DELETED_FRAMES:
            frame = video_path + '/' + frame
            features = extract_features(model, frame)
            sequence.append(features)
            if len(sequence) == SEQ_LENGTH:
                X.append(sequence)
                y.append(EMOTIONS.index(emotion))
                # no overlapping frames if sequence length is less than 2
                if SEQ_LENGTH > 1:
                    sequence = sequence[OVERLAP_IDX:]
                else:
                    sequence = []
    return X, y

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
    model = VGG16(include_top=False, weights='imagenet')
    extract_feature_sequence(model)
    print('Sequences saved')