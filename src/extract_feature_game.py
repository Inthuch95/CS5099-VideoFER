'''
Created on Aug 2, 2018

@author: Inthuch Therdchanakul    
'''
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
import cv2
import dlib
import pickle
import os

DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
EMOTIONS = DATA['EMOTIONS'] 
IMG_WIDTH, IMG_HEIGHT = 100,100
SEQ_LENGTH = 2
OVERLAP_IDX = int(0.9 * SEQ_LENGTH)
DATA_PATH = '../prepared_data/Game/data/'
EXTRACT_PATH = '../prepared_data/Game/frames/'
SEQUENCE_PATH = '../prepared_data/Game/sequence/'

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
    print(np.array(X).shape)
    np.save(SEQUENCE_PATH+'X_game.npy', X)
    np.save(SEQUENCE_PATH+'y_game.npy', y)

def process_frames(frames, video_path, emotion, X, y):
    sequence = []      
    for frame in frames:
        # exclude neutral frames 
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

def crop_face_from_frames():
    count = 0
    face_detector = dlib.get_frontal_face_detector()
    
    for emotion in EMOTIONS:
        print(emotion)
        if not os.path.exists(DATA_PATH + emotion):
            os.mkdir(DATA_PATH + emotion)
        for frame_dir in os.listdir(os.path.join(EXTRACT_PATH, emotion)):
            frame_path = os.path.join(EXTRACT_PATH, emotion, frame_dir)
            save_path = os.path.join(DATA_PATH, emotion, frame_dir)
            if not os.path.exists(save_path):
                os.mkdir(save_path) 
            filelist = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]
            # Iterate through files
            for f in filelist:
                try:
                    frames = 0
                    vidcap = cv2.VideoCapture(os.path.join(frame_path, f))
                    framecount = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                    while frames < framecount:
                        _, frame = vidcap.read()
                        # detect face
                        detected_face = face_detector(frame, 1)
                        # crop and save detected face
                        if len(detected_face) > 0:
                            save_frame(detected_face, frame, save_path, count, f)
                            count +=1
                            frames += 1
                except RuntimeError:
                    continue
                
def save_frame(detected_face, frame, save_path, count, filename):
    for _, d in enumerate(detected_face):
        crop = frame[d.top():d.bottom(), d.left():d.right()]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # save frame as JPEG file
        cv2.imwrite(save_path+'/'+filename, crop)

if __name__ == '__main__':
#     crop_face_from_frames()
    model = VGG16(include_top=False, weights='imagenet')
    extract_feature_sequence(model)
    print('Sequences saved')