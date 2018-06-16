'''
Created on Jun 16, 2018

@author: Inthuch Therdchanakul
'''
import os

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
video_path =  '../EUDataBasicVideo/'

def extract_frames_from_video():
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
    
if __name__ == '__main__':
    extract_frames_from_video()