'''
Created on Jun 26, 2018

@author: User
'''
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image
import os

img_width, img_height = 100,100
seq_length = 3
frame_path = '../dataset2/All/'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotionLbls = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0], [0,0,1,0,0,0,0],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

def get_feature_sequence(model):
    for emotion in emotions:
        video_list = [f for f in os.listdir(frame_path + emotion) if '.npy' not in f]
        for video in video_list:
            video_path = frame_path + emotion + '/' + video
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
                        path = frame_path + emotion + '/' + filename
                        np.save(path, sequence)
                        sequence = []
                        index += 1
        print('{} sequences extracted'.format(emotion))

def save_feature_sequence():
    X_train, y_train = [],[] 
    X_test, y_test = [],[]
    for emotion in emotions:
        sequence_path = frame_path + emotion
        extracted_seq = [f for f in os.listdir(sequence_path) if '.npy' in f]
        split_index = int(0.9 * len(extracted_seq)) 
        for f in extracted_seq[:split_index]:
            sequence = np.load(frame_path + emotion + '/' + f)
            X_train.append(sequence)
            y_train.append(emotionLbls[emotions.index(emotion)])
        for f in extracted_seq[split_index:]:
            sequence = np.load(frame_path + emotion + '/' + f)
            X_test.append(sequence)
            y_test.append(emotionLbls[emotions.index(emotion)])
        print('{} processed'.format(emotion))
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)
    np.save('../dataset2/X_train_inception.npy', X_train)
    np.save('../dataset2/y_train_inception.npy', y_train)
    
    X_test = np.array(X_test)
    print(X_test.shape)
    y_test = np.array(y_test)
    np.save('../dataset2/X_test_inception.npy', X_test)
    np.save('../dataset2/y_test_inception.npy', y_test)

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
    base_model = InceptionV3(include_top=True, weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    get_feature_sequence(model)
    save_feature_sequence()
    print('Sequences saved')