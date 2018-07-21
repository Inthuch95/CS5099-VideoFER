'''
Created on Jul 18, 2018

@author: Inthuch Therdchanakul
'''
import os
import subprocess

command = '../OpenFace_2.0.3_win_x64/FeatureExtraction.exe'
sequence_dir = '../prepared_data/Basic/data/Angry/FF17/'

def test_au():
    p = subprocess.Popen([command, '-fdir', sequence_dir, '-aus'], cwd='../OpenFace_2.0.3_win_x64/')
    p.wait()

if __name__ == '__main__':
    test_au()