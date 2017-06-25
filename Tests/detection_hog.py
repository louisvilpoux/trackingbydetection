import numpy as np
import pandas as pd
import cv2
import imutils

video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/people_walking_short.mp4"

min_area = 100
winStride = (8, 8)
padding = (16, 16)
scale = 1.05
meanShift = True
cap = cv2.VideoCapture(video)
#cap.set(3, 640)
#cap.set(4, 480)
fgbg = cv2.BackgroundSubtractorMOG2()
#fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    fgmask = fgbg.apply(frame)

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=padding, scale=scale, 
        useMeanshiftGrouping=meanShift)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)
    #cv2.imshow('thresh',thresh)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()