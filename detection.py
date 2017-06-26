#cd Documents/Manchester/Dissertation/trackingbydetection/

import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

save_paticles = []
save_detections = []
colors = {"red" : (255, 0, 0), "green" : (0, 255, 0), "white" : (255, 255, 255), 
          "blue" : (0, 0, 255), "yellow" : (255, 255, 0) , "turquoise" : (0, 255, 255), "purple" : (255, 0, 255)}
number_particles = 100

#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/people-walking.mp4"
video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/mot1.mp4"

# minimum size of the contours that will be considered. It permits to not deal with very little detections (noise)
min_area = 50

nb = 0
val = 1000

cap = cv2.VideoCapture(video)
#fgbg = cv2.BackgroundSubtractorMOG2()
fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    #resize because of the performance
    frame = imutils.resize(frame, width=500)

    #learning rate set to 0
    #fgmask = fgbg.apply(frame)
    fgmask = fgbg.apply(frame, learningRate=1.0/10)
    
    (cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
 
		# compute the bounding box for the contour, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # compute the center of the contour for each detection. cX and cY are the coords of the detection center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 2, (255, 255, 255), -1)
        cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # draw the particles
        # find the optimal size limit for the particles spread
        mean = [cX, cY]
        val_cov = min(w,h)
        cov = [[val_cov, 0], [0, val_cov]]
        part_x, part_y = np.random.multivariate_normal(mean, cov, number_particles).T
        # plot the particles
        #plt.plot(part_x, part_y, 'x')
        #plt.axis('equal')
        #plt.show()

        # build the matrix of the detections respecting data model
        # detection : x_center, y_center, x_min_contour, y_min_contour, x_max_contour, y_max_contour
        save_detections.append([cX,cY,x,y,x+w,y+h])

        # build the matrix of the particles respecting data model
        # particle : x_ord, y_ord, weight, x_detection_center, y_detection_center, frame_count_since_born
        weight = 0
        frame_born = 0
        for i,j in zip(part_x,part_y):
			save_paticles.append([i,j,weight,None,None,frame_born])

        # Print the data of a special frame
        # nb = nb + 1
        # if nb == 1:
        #     print("part", save_paticles)

    # Calculate the distance between each detection,particle pair.
    for detect in save_detections:
        for particl in save_paticles:
            d = [detect[0],detect[1]]
            p = [particl[0],particl[1]]
            norm_d_p = ssp.distance.euclidean(d,p)

 
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)
    #cv2.imshow('thresh',thresh)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
