'''
Created on Jun 16, 2018

@author: Inthuch Therdchanakul
'''
import os
import cv2
import dlib
from timeit import default_timer as timer

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def extract_frames_from_video():
    video_path =  '../EUDataBasicVideo/'
    if 'video_frames' not in os.listdir('../'):
        os.mkdir('../video_frames/')
    #go through video folder
    for emotion in emotions:
        # create label folder
        newdir = '../video_frames/' + emotion
        if emotion not in os.listdir('../video_frames/'):
            os.makedirs(newdir)
        for f in os.listdir(video_path+emotion):
            if '.mov' in f:
                filename = f.replace('.mov', '')
                video_file = video_path + emotion + '/' + f
                filename = '../video_frames/' + emotion +  '/' + filename
                # use ffmpeg to extract frames
                command = 'ffmpeg -i ' + video_file + ' -vf thumbnail=2,setpts=N/TB -r 1 -vframes 300 ' + filename + '%05d.jpeg'
                os.system(command)
                
def crop_face_from_frames():
    path = '../video_frames/'
    dataset_path = '../dataset/'
    count = 0 
    face_detector = dlib.get_frontal_face_detector()
    
    if 'dataset' not in os.listdir('../'):
        os.mkdir(dataset_path)
    for emotion in emotions:
        start = timer()
        print(emotion)
        if emotion not in os.listdir(dataset_path):
            os.mkdir(dataset_path + emotion)
        frame_path = os.path.join(path, emotion)
        save_path = os.path.join(dataset_path, emotion) 
        # Iterate through files
        filelist = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]
        index = 0
        for f in filelist:
            if index % 100 == 0:
                percentage = (index / len(filelist)) * 100
                checkpoint = timer()
                elapsed_time = int(checkpoint - start)
                print('{}% complete, elapsed time {}s'.format(str(percentage), str(elapsed_time)))
            index += 1
            try:
                frames = 0
                vidcap = cv2.VideoCapture(os.path.join(frame_path, f))
                framecount = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                while frames < framecount:
                    _, frame = vidcap.read()
#                     print(f)
                    # detect face
                    detected_face = face_detector(frame, 1)
    #                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # crop and save detected face
                    if len(detected_face) > 0:
#                         print("Number of faces detected: {}".format(len(detected_face)))
                        for i, d in enumerate(detected_face):
#                             print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#                                 i, d.left(), d.top(), d.right(), d.bottom()))
        #                     frame = gray
                            crop = frame[d.top():d.bottom(), d.left():d.right()]
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            # save frame as JPEG file
                            cv2.imwrite(save_path+'/frame%d.jpg' % count, crop)  
                            count +=1
                            frames += 1
            except RuntimeError:
                continue
    end = timer()
    total_time = int(end - start)
    print('Elapsed time {}s'.format(total_time))   
    
if __name__ == '__main__':
#     extract_frames_from_video()
    crop_face_from_frames()