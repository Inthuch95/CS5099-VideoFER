'''
Created on Jun 26, 2018

@author: User
'''
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import os

img_width, img_height = 100,100
seq_length = 20
frame_path = '../dataset2/All/'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotionLbls = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0], [0,0,1,0,0,0,0],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

def get_feature_sequence(model):
    for emotion in emotions:
        video_list = [f for f in os.listdir(frame_path + emotion) if '.npy' not in f]
        for video in video_list:
            video_path = frame_path + emotion + '/' + video
            frames = get_frames(video_path)
            frames = rescale_list(frames, seq_length)
            sequence = []
            filename = video + '_feature.npy'
            path = frame_path + emotion + '/' + filename
            for frame in frames:
                frame = video_path + '/' + frame
                features = extract_features(model, frame)
                sequence.append(features)
            # Save the sequence.)
            np.save(path, sequence)

def save_feature_sequence():
    X_train, y_train = [],[] 
    X_test, y_test = [],[]
    for emotion in emotions:
        sequence_path = frame_path + emotion
        extracted_seq = [f for f in os.listdir(sequence_path) if '.npy' in f] 
        for f in extracted_seq[:-2]:
            sequence = np.load(frame_path + emotion + '/' + f)
            X_train.append(sequence)
            y_train.append(emotionLbls[emotions.index(emotion)])
        for f in extracted_seq[-2:]:
            sequence = np.load(frame_path + emotion + '/' + f)
            X_test.append(sequence)
            y_test.append(emotionLbls[emotions.index(emotion)])
        print('{} processed'.format(emotion))
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    np.save('../dataset2/X_train.npy', X_train)
    np.save('../dataset2/y_train.npy', y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    np.save('../dataset2/X_test.npy', X_test)
    np.save('../dataset2/y_test.npy', y_test)

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

def rescale_list(input_list, size):
    assert len(input_list) >= size
    # Get the number to skip between iterations.
    skip = len(input_list) // size
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    # Cut off the last one if needed.
    return output[:size]

if __name__ == '__main__':
#     model = VGG16(include_top=False, weights='imagenet')
#     get_feature_sequence(model)
    save_feature_sequence()
    print('Sequences saved')