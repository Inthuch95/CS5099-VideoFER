'''
Created on Jun 27, 2018

@author: User
'''
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import random
import os

img_width, img_height = 100,100
seq_length = 2
data_path = '../prepared_data/All/'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotionLbls = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0], [0,0,1,0,0,0,0],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

# get sequence of features for RNN
def extract_feature_sequence(model):
    for emotion in emotions:
        if 'vgg16' not in os.listdir(data_path + emotion):
            os.mkdir(data_path + emotion + '/' + 'vgg16/')
        video_list = [f for f in os.listdir(data_path + emotion) if 'vgg16' not in f]
        for video in video_list:
            video_path = data_path + emotion + '/' + video
            frames = get_frames(video_path)
            if len(frames) >= seq_length:
                sequence = []
                index = 0
                for frame in frames:
                    frame = video_path + '/' + frame
                    features = extract_features(model, frame)
                    sequence.append(features)
                    if len(sequence) == seq_length:
                        # Save the sequence.)
                        filename = video + '_feature_' + str(index) + '.npy'
                        path = data_path + emotion + '/vgg16/' + filename
                        np.save(path, sequence)
                        sequence = []
                        index += 1
        print('{} sequences extracted'.format(emotion))

def save_feature_sequence():
    X_train, X_val, y_train = [], [], [] 
    X_test, y_val, y_test = [], [], []
    for emotion in emotions:
        sequence_path = data_path + emotion + '/vgg16/'
        print()
        extracted_seq = [f for f in os.listdir(sequence_path) if '.npy' in f]
        random.seed(42)
        random.shuffle(extracted_seq)
        split_1 = int(0.8 * len(extracted_seq)) 
        split_2 = int(0.9 * len(extracted_seq)) 
        for f in extracted_seq[:split_1]:
            sequence = np.load(sequence_path  + f)
            X_train.append(sequence)
            y_train.append(emotionLbls[emotions.index(emotion)])
        for f in extracted_seq[split_1:split_2]:
            sequence = np.load(sequence_path  + f)
            X_val.append(sequence)
            y_val.append(emotionLbls[emotions.index(emotion)])  
        for f in extracted_seq[split_2:]:
            sequence = np.load(sequence_path  + f)
            X_test.append(sequence)
            y_test.append(emotionLbls[emotions.index(emotion)])
        print('{} processed'.format(emotion))
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)
    np.save('../prepared_data/sequence/X_train_vgg16.npy', X_train)
    np.save('../prepared_data/sequence/y_train_vgg16.npy', y_train)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print(X_val.shape)
    np.save('../prepared_data/sequence/X_val_vgg16.npy', X_val)
    np.save('../prepared_data/sequence/y_val_vgg16.npy', y_val)
    
    X_test = np.array(X_test)
    print(X_test.shape)
    y_test = np.array(y_test)
    np.save('../prepared_data/sequence/X_test_vgg16.npy', X_test)
    np.save('../prepared_data/sequence/y_test_vgg16.npy', y_test)

# extract features from individual frame (for other classifiers like SVM)        
def extract_feature_single(model):
    X, y = [], []
    for emotion in emotions:
        video_list = [f for f in os.listdir(data_path + emotion) if 'vgg16' not in f]
        for video in video_list:
            video_path = data_path + emotion + '/' + video
            frames = get_frames(video_path)
            for frame in frames:
                frame = video_path + '/' + frame
                features = extract_features(model, frame)
                X.append(features)
                y.append(emotion)
        # Save the sequence
        print('{} features extracted'.format(emotion))
    np.save('../prepared_data/single/X_vgg16.npy', X)
    np.save('../prepared_data/single/y_vgg16.npy', y) 

def save_feature_single():
    X = np.load('../prepared_data/single/X_vgg16.npy')
    y = np.load('../prepared_data/single/y_vgg16.npy')
    data = list(zip(X, y))
    random.seed(42)
    random.shuffle(data)
    X, y = zip(*data)
    split_index = int(0.8 * len(X)) 
    
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)
    np.save('../prepared_data/single/X_train_vgg16.npy', X_train)
    np.save('../prepared_data/single/y_train_vgg16.npy', y_train)
    
    X_test = X[split_index:]
    y_test = y[split_index:]
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(X_test.shape)
    np.save('../prepared_data/single/X_test_vgg16.npy', X_test)
    np.save('../prepared_data/single/y_test_vgg16.npy', y_test) 

def extract_features(model, image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the prediction.
    features = model.predict(x)
    features = features[0]
    return features

def get_frames(path):
    frames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return frames

if __name__ == '__main__':
    model = VGG16(include_top=False, weights='imagenet')
    extract_feature_sequence(model)
    save_feature_sequence()  
#     extract_feature_single(model)
#     save_feature_single()
    print('Sequences saved')