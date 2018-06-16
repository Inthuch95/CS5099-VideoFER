'''
Created on Jun 16, 2018

@author: User
'''
import os
import cv2
import dlib

image_path = '../video_frames/Fear'

if __name__ == '__main__':
    face_cascade = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(face_cascade)
    count = 0 
    detector = dlib.get_frontal_face_detector()
    # Iterate through files
    for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
#         save_faces(cascade, f)
        try:
            frames = 0
            vidcap = cv2.VideoCapture(image_path+ '/' + f)
            framecount = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            #vidcap.open(image_path + emotion+'/'+filename)
            while frames < framecount: #(vidcap.isOpened()):
                ret, frame = vidcap.read()
                print(f)
                dets = detector(frame, 1)
                if len(dets) > 0:
                    print("Number of faces detected: {}".format(len(dets)))
                    for i, d in enumerate(dets):
                        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                            i, d.left(), d.top(), d.right(), d.bottom()))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                     img = gray
                    img = frame
                    crop = img[d.top():d.bottom(), d.left():d.right()]
                    #cv2.imshow('frame',gray)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    cv2.imwrite('../video_frames/faces/'+'frame%d.jpg' % count, crop)  # save frame as JPEG file
                    count +=1
                    frames += 1
        except RuntimeError:
            continue   