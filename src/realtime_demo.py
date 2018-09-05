'''
Created on Jul 12, 2018

@author: Inthuch Therdchanakul
'''
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import pandas as pd
import subprocess
from utils import load_var
import numpy as np
import cv2
import dlib
import os

def process_vgg16_img(img_file, sequence):
    img = image.img_to_array(img_file)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    sequence.append(img)
    return sequence

def extract_vgg16_feature(vgg16, sequence):
    for i in range(len(sequence)):
        features = vgg16.predict(sequence[i])
        sequence[i] = features[0]
    X = np.array([sequence])
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4])
    return X

def extract_au_feature():
    command = '../OpenFace_2.0.3_win_x64/FeatureExtraction.exe -fdir ../temp/ -aus -out_dir ../temp/' 
    p = subprocess.Popen(command.split(), cwd='../OpenFace_2.0.3_win_x64/')
    p.wait()
    file = [f for f in os.listdir('../temp/') if '.csv' in f]
    au_file = file[0]
    df = pd.read_csv('../temp/' + au_file)
    au_col = [col for col in df.columns if 'AU' in col and '_r' in col]
    sequence = []
    for _, frame in df.iterrows():
        au_vals = [val for val in frame[au_col]]
        sequence.append(au_vals)
    X = np.array([sequence])
    return X

def clean_temp_dir():
    folder = '../temp/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


if not os.path.exists('../temp/'):
    os.mkdir('../temp/')

data_type = 'Basic'
video_capture = cv2.VideoCapture(0)
frameWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
feature = 'VGG16'
SEQ_LENGTH = 2
DATA = load_var(data_type)
EMOTIONS = DATA['EMOTIONS']
model_file =  '../best model/' + data_type + '/' + feature + '/LSTM.h5'
model = load_model(model_file)
vgg16 = VGG16(include_top=False, weights='imagenet')
face_detector = dlib.get_frontal_face_detector()
emotion =''
count = 0
frame_count = 0
sequence = []
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    faces = face_detector(frame, 1)
    
    if ret != 0 and len(faces) != 0:
#         if frame_count % 2 == 0:
        # Draw a rectangle around the faces
        crop = frame[faces[0].top():faces[0].bottom(), faces[0].left():faces[0].right()]
        cv2.imwrite('../temp/frame%d.jpg' % count, crop)
        try:
            img_file = image.load_img('../temp/frame%d.jpg' % count, target_size=(100, 100))
        except OSError:
            continue
        if feature == 'VGG16':
            sequence = process_vgg16_img(img_file, sequence)
        elif feature == 'AU':
            sequence.append(0)
        if len(sequence) == SEQ_LENGTH:
            if feature == 'VGG16':
                X = extract_vgg16_feature(vgg16, sequence)
            elif feature == 'AU':
                X = extract_au_feature()
            prediction = model.predict(X)
            pred = list(prediction[0])
            max_index = np.argmax(pred)
            emotion = EMOTIONS[max_index]
            sequence = []
            clean_temp_dir()
            
        cv2.rectangle(frame, (faces[0].left(), faces[0].top()), (faces[0].right(), faces[0].bottom()), (0, 255, 0), 2)
        cv2.putText(img = frame, text = emotion, org = (int(frameWidth/2 - 20),int(frameHeight/2)), 
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 2, color = (0, 255, 0))
        frame_count  += 1 
    # Display the resulting frame
    cv2.imshow('Video', frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
clean_temp_dir()