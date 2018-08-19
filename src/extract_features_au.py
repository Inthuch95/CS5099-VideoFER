'''
Created on Jul 18, 2018

@author: Inthuch Therdchanakul
'''
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import pickle
import sys
import os
import subprocess

data_type = 'Basic'
if data_type == 'Basic':
    DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
    au_path = '../prepared_data/Basic/au_data/'
elif data_type == 'Complex':
    DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
    au_path = '../prepared_data/Complex/au_data/'
else:
    print("Invalid data type")
    sys.exit()
IMG_WIDTH, IMG_HEIGHT = 100,100
SEQ_LENGTH = 2
OVERLAP_IDX = int(0.9 * SEQ_LENGTH)
DATA_PATH = DATA['DATA_PATH']
EMOTIONS = DATA['EMOTIONS']
DELETED_FRAMES = DATA['DELETED_FRAMES']
SEQUENCE_PATH = DATA['SEQUENCE_PATH']
SINGLE_PATH = DATA['SINGLE_PATH']
    
def extract_au():
    for emotion in EMOTIONS:
        video_list = [f for f in os.listdir(DATA_PATH + emotion)]
        out_dir = au_path + emotion
        for video in video_list:
            sequence_dir = DATA_PATH + emotion + '/' + video
            if len(os.listdir(sequence_dir)) >= SEQ_LENGTH:
                # extract AU with OpenFace
                # these are saved in csv files
                command = '../OpenFace_2.0.3_win_x64/FeatureExtraction.exe -fdir ' + sequence_dir + ' -aus -out_dir ' + out_dir 
                p = subprocess.Popen(command.split(), cwd='../OpenFace_2.0.3_win_x64/')
                p.wait()
        print('{} sequences extracted'.format(emotion))
        
def build_au_sequence():
    X, y = [], []
    # put extracted AUs into sequence for LSTM 
    for emotion in EMOTIONS:
        extracted_au = [f for f in os.listdir(au_path + emotion) if '.csv' in f]
        for f in extracted_au:
            path = au_path + emotion + '/' + f
            df = pd.read_csv(path)
            X, y = get_sequence(X, y, df, emotion)
    if SEQ_LENGTH > 1:
        y = to_categorical(y, num_classes=len(EMOTIONS))
    # save to binary files
    print('Saving sequence')
    if SEQ_LENGTH == 1:
        np.save(SINGLE_PATH+'X_au.npy', X)
        np.save(SINGLE_PATH+'y_au.npy', y)
    else:
        # data for SVM
        np.save(SEQUENCE_PATH+'X_au.npy', X)
        np.save(SEQUENCE_PATH+'y_au.npy', y)

def get_sequence(X, y, df, emotion):
    au_col = [col for col in df.columns if 'AU' in col and '_r' in col]
    sequence = []
    for _, frame in df.iterrows():
        if frame[' success'] != 0:
            au_vals = [val for val in frame[au_col]]
            sequence.append(au_vals)
            if len(sequence) == SEQ_LENGTH:
                X.append(sequence)
                y.append(EMOTIONS.index(emotion))
                # no overlapping frames if sequence length is less than 2
                if SEQ_LENGTH > 1:
                    sequence = sequence[OVERLAP_IDX:]
                else:
                    sequence = []
    return X, y
                
if __name__ == '__main__':
    extract_au()
    build_au_sequence()