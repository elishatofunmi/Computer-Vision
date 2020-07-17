from imutils.video import VideoStream
from imutils.video import FPS
import tensorflow as tf
import keras 
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from keras.models import load_model

model = load_model('pure_model.h5')


cam = cv2.VideoCapture(0)
print("[INFO] starting video stream...")
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=800)
     
    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]

        roi = cv2.resize(fc, (224,224,3))
        pred = model.predict(roi)
        value = np.argmax(pred,axis = 1)
        if value == 0:
            cv2.putText(fc, 'tomatoes', (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fc,(x,y),(x+w,y+h),(255,0,0),2)
        
        else:
            cv2.put
    cv2.imshow('object detection', output_image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    # update the FPS counter
    fps.update()
    fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()