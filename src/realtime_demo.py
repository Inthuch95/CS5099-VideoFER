'''
Created on Jul 12, 2018

@author: Inthuch Therdchanakul
'''
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pickle
import cv2
import dlib
import os
import sys

data_type = 'Basic'
SEQ_LENGTH = 2
# data_type = 'Complex'
if data_type == 'Basic':
    DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
elif data_type == 'Complex':
    DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
else:
    print("Invalid data type")
    sys.exit()
EMOTIONS = DATA['EMOTIONS']
base_dir = '../best model/' + data_type + '/'
model_file =  base_dir + 'LSTM_best/LSTM_best.h5'
model = load_model(model_file)
vgg16 = VGG16(include_top=False, weights='imagenet')
face_detector = dlib.get_frontal_face_detector()

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

emotion =''
video_capture = cv2.VideoCapture(0)
frameWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = 0
frame_count = 0
sequence = []
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    faces = face_detector(frame, 1)
    
    if ret != 0 and len(faces) != 0:
        if frame_count % 3 == 0:
            # Draw a rectangle around the faces
            crop = frame[faces[0].top():faces[0].bottom(), faces[0].left():faces[0].right()]
            cv2.imwrite('../temp/frame%d.jpg' % count, crop)
            try:
                img = image.load_img('../temp/frame%d.jpg' % count, target_size=(100, 100))
            except OSError:
                continue
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            sequence.append(x)
            if len(sequence) == SEQ_LENGTH:
                for i in range(len(sequence)):
                    features = vgg16.predict(sequence[i])
                    sequence[i] = features[0]
                X = np.array([sequence])
                X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4])
                prediction = model.predict(X)
                pred = list(prediction[0])
                max_value = max(pred)
                max_index = pred.index(max_value)
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