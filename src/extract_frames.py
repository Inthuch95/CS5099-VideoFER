'''
Created on Jun 16, 2018

@author: Inthuch Therdchanakul
'''
import os

emotions = [ 'Angry', 'Disgust', 'Fear', 'Neutral', 'Sad', 'Surprise']
video_path =  '../EUDataBasicVideo/'

def capture_images():
    #go through video folder
    for emotion in emotions:
        newdir = '../video_frames/' + emotion
        if emotion not in os.listdir('../video_frames/'):
            os.makedirs(newdir)
        for filename in os.listdir(video_path+emotion):
            if '.mov' in filename:
                filename1 = filename
                filename = filename.replace('.mov', '')
                file = video_path + emotion + '/' + filename1
                filename = '../video_frames/' + emotion +  '/' + filename
                command = 'ffmpeg -i ' + file + ' -vf thumbnail=2,setpts=N/TB -r 1 -vframes 300 ' + filename + '%05d.jpeg'
                os.system(command)
if __name__ == '__main__':
    capture_images()