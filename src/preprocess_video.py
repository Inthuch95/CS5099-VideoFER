'''
Created on Jun 26, 2018

@author: Inthuch Therdchanakul
'''
import os
import cv2
import dlib

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def extract_frames_from_video():
    video_path =  '../EUDataBasicVideo/'
    if not os.path.exists('../video_frames/'):
        os.mkdir('../video_frames/')
        os.mkdir('../video_frames/Basic/')
    #go through video folder
    for emotion in emotions:
        # create label folder
        emotion_path = '../video_frames/Basic/' + emotion
        if not os.path.exists(emotion_path):
            os.mkdir(emotion_path)
        for f in os.listdir(video_path+emotion):
            if '.mov' in f:
                filename = f.replace('.mov', '')
                video_file = video_path + emotion + '/' + f
                os.mkdir('../video_frames/Basic/' + emotion + '/' + filename +  '/')
                filename = '../video_frames/Basic/' + emotion + '/' + filename +  '/' + filename
                # use ffmpeg to extract frames
                command = 'ffmpeg -i ' + video_file + ' -vf thumbnail=2,setpts=N/TB -r 1 -vframes 300 ' + filename + '%05d.jpeg'
                os.system(command)
                
def crop_face_from_frames():
    path = '../video_frames/Basic/'
    dataset_path = '../prepared_data/Basic/data/'
    count = 0 
    face_detector = dlib.get_frontal_face_detector()
    
    for emotion in emotions:
        print(emotion)
        if not os.path.exists(dataset_path + emotion):
            os.mkdir(dataset_path + emotion)
        for frame_dir in os.listdir(os.path.join(path, emotion)):
            frame_path = os.path.join(path, emotion, frame_dir)
            save_path = os.path.join(dataset_path, emotion, frame_dir)
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
    extract_frames_from_video()
    crop_face_from_frames()