'''
Created on Jun 16, 2018

@author: Inthuch Therdchanakul
'''
import os
import cv2
import dlib
from timeit import default_timer as timer
from shutil import copyfile, rmtree

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
    dataset_path = '../dataset/All/'
    count = 0 
    face_detector = dlib.get_frontal_face_detector()
    
    if 'dataset' not in os.listdir('../'):
        os.mkdir(dataset_path)
    start = timer()
    for emotion in emotions:
        start_emotion = timer()
        print(emotion)
        if emotion not in os.listdir(dataset_path):
            os.mkdir(dataset_path + emotion)
        frame_path = os.path.join(path, emotion)
        save_path = os.path.join(dataset_path, emotion) 
        filelist = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]
        index = 0
        # Iterate through files
        for f in filelist:
            # print progress every 100 frames
            if index % 100 == 0:
                print_progress(index, filelist, start_emotion)
            index += 1
            try:
                frames = 0
                vidcap = cv2.VideoCapture(os.path.join(frame_path, f))
                framecount = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                while frames < framecount:
                    _, frame = vidcap.read()
                    #print(f)
                    # detect face
                    detected_face = face_detector(frame, 1)
                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # crop and save detected face
                    if len(detected_face) > 0:
                        #print("Number of faces detected: {}".format(len(detected_face)))
                        count, frames = save_frame(detected_face, frame, save_path, count, frames)
            except RuntimeError:
                continue
    end = timer()
    total_time = int(end - start)
    print('Elapsed time {}s'.format(total_time))
    
def save_frame(detected_face, frame, save_path, count, frames):
    for _, d in enumerate(detected_face):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             i, d.left(), d.top(), d.right(), d.bottom()))
#             frame = gray
        crop = frame[d.top():d.bottom(), d.left():d.right()]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # save frame as JPEG file
        cv2.imwrite(save_path+'/frame%d.jpg' % count, crop)  
        count +=1
        frames += 1
    return count, frames

def print_progress(index, filelist, start):
    percentage = (index / len(filelist)) * 100
    checkpoint = timer()
    elapsed_time = int(checkpoint - start)
    print('{}% complete, elapsed time {}s'.format(str(percentage), str(elapsed_time)))
    
def train_test_split():
    dataset_path = '../dataset/All/'
    train_path = '../dataset/Train/'
    val_path = '../dataset/Validation/'
    test_path = '../dataset/Test/'
    
    delete_data(train_path, val_path, test_path)
    # create folders for train_lstm/val/test data
    create_data_dir(train_path, val_path, test_path)
    create_label_dir(train_path, val_path, test_path, emotions)
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        filelist = [f for f in os.listdir(emotion_path) if os.path.isfile(os.path.join(emotion_path, f))]
        # define split index
        # 80% train, 10% validation, 10% test
        train_split = int(0.8 * len(filelist))
        val_split = int(0.9 * len(filelist))
        
        # split the data
        train_files = filelist[:train_split]
        val_files = filelist[train_split:val_split]
        test_files = filelist[val_split:]
        
        # copy files into folders
        for f in train_files:
            src = os.path.join(emotion_path, f)
            dst = os.path.join(train_path, emotion, f)
            copyfile(src, dst)
            
        for f in val_files:
            src = os.path.join(emotion_path, f)
            dst = os.path.join(val_path, emotion, f)
            copyfile(src, dst)
            
        for f in test_files:
            src = os.path.join(emotion_path, f)
            dst = os.path.join(test_path, emotion, f)
            copyfile(src, dst)
        print('{} data saved'.format(emotion))

def delete_data(train_path, val_path, test_path):
    # delete previous data
    try:
        if 'Train' not in os.listdir('../dataset/'):
            rmtree(train_path)
        if 'Validation' not in os.listdir('../dataset/'):
            rmtree(val_path)
        if 'Test' not in os.listdir('../dataset/'):
            rmtree(test_path)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
            
def create_data_dir(train_path, val_path, test_path):
    # create folders for train/validation/test data
    try:
        if 'Train' not in os.listdir('../dataset/'):
            os.mkdir(train_path)
        if 'Validation' not in os.listdir('../dataset/'):
            os.mkdir(val_path)
        if 'Test' not in os.listdir('../dataset/'):
            os.mkdir(test_path)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        
def create_label_dir(train_path, val_path, test_path, labels):
    # create folders for each label
    try:
        for label in labels:
            os.mkdir(os.path.join(train_path, label))
            os.mkdir(os.path.join(val_path, label))
            os.mkdir(os.path.join(test_path, label))
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    
if __name__ == '__main__':
    extract_frames_from_video()
    crop_face_from_frames()
    train_test_split()  