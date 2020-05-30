import cv2
import sys
from time import sleep

cam = cv2.VideoCapture(0)
cam.set(3, 1920)
cam.set(4, 1080)
cascPath = "face.xml"
ret, frame = cam.read()
frame_last = frame
while True:
    try:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        faceCascade = cv2.CascadeClassifier(cascPath)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize= (15,15),
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        
        print("Found {0} Faces!".format(len(faces)))
        for (x, y, w, h,) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow('Webcam | Live Stream', frame)
        cv2.waitKey(4)
        #sleep(0.1)
    except KeyboardInterrupt:
        break
cam.release()
