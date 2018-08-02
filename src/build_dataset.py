'''
Created on Jun 26, 2018

@author: Inthuch Therdchanakul
'''
import os
import cv2
import dlib
import pickle
import sys

data_type = 'Basic'
if data_type == 'Basic':
    DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
elif data_type == 'Complex':
    DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
else:
    print("Invalid data type")
    sys.exit()
EMOTIONS = list(DATA['EMOTIONS'])
VIDEO_PATH =  DATA['VIDEO_PATH']
EXTRACT_PATH = DATA['EXTRACT_PATH']
DATA_PATH = DATA['DATA_PATH']

def extract_frames_from_video():
    if not os.path.exists('../video_frames/'):
        os.mkdir('../video_frames/')
        os.mkdir(EXTRACT_PATH)
    #go through video folder
    for emotion in EMOTIONS:
        # create label folder
        emotion_path = EXTRACT_PATH + emotion
        if not os.path.exists(emotion_path):
            os.mkdir(emotion_path)
        for f in os.listdir(VIDEO_PATH+emotion):
            if '.mov' in f or 'mp4' in f:
                if '.mov' in f:
                    filename = f.replace('.mov', '')
                else:
                    filename = f.replace('.mp4', '')
                video_file = VIDEO_PATH + emotion + '/' + f
                if not os.path.exists(emotion_path + '/' + filename +  '/'):
                    os.mkdir(emotion_path + '/' + filename +  '/')
                file_path = emotion_path + '/' + filename +  '/' + filename
                # use ffmpeg to extract frames
                command = 'ffmpeg -i ' + video_file + ' -vf thumbnail=2,setpts=N/TB -r 1 -vframes 300 ' + file_path + '%05d.jpeg'
                os.system(command)
                
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
                            save_frame(detected_face, frame, save_path, count)
                            count +=1
                            frames += 1
                except RuntimeError:
                    continue

def save_frame(detected_face, frame, save_path, count):
    for _, d in enumerate(detected_face):
        crop = frame[d.top():d.bottom(), d.left():d.right()]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # save frame as JPEG file
        cv2.imwrite(save_path+'/frame%d.jpg' % count, crop)

if __name__ == '__main__':
    pass
#     extract_frames_from_video()
#     crop_face_from_frames()