'''
Created on Jun 16, 2018

@author: Inthuch Therdchanakul
'''
import os
import cv2
import dlib

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
frame_path = '../video_frames/Fear'
count = 0 
detector = dlib.get_frontal_face_detector()

if __name__ == '__main__':
    # Iterate through files
    filelist = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]
    for f in filelist:
        try:
            frames = 0
            vidcap = cv2.VideoCapture(frame_path + '/' + f)
            framecount = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            while frames < framecount:
                ret, frame = vidcap.read()
                print(f)
                # detect face
                detected_face = detector(frame, 1)
                # crop and save detected face
                if len(detected_face) > 0:
                    print("Number of faces detected: {}".format(len(detected_face)))
                    for i, d in enumerate(detected_face):
                        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                            i, d.left(), d.top(), d.right(), d.bottom()))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     img = gray
                    img = frame
                    crop = img[d.top():d.bottom(), d.left():d.right()]
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    # save frame as JPEG file
                    cv2.imwrite('../video_frames/faces/'+'frame%d.jpg' % count, crop)  
                    count +=1
                    frames += 1
        except RuntimeError:
            continue   