import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

save_paticles = []
save_centers = []
colors = {"red" : (255, 0, 0), "green" : (0, 255, 0), "white" : (255, 255, 255), 
          "blue" : (0, 0, 255), "yellow" : (255, 255, 0) , "turquoise" : (0, 255, 255), "purple" : (255, 0, 255)}

#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/people-walking.mp4"
video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/mot1.mp4"

min_area = 50

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

        # compute the center of the contour for each detection
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 2, (255, 255, 255), -1)
        cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # draw the particles
        mean = [cX, cY]
        cov = [[10, 0], [0, 10]]
        number_particles = 100
        part_x, part_y = np.random.multivariate_normal(mean, cov, number_particles).T
        # plot the particles
        #plt.plot(part_x, part_y, 'x')
        #plt.axis('equal')
        #plt.show()

        # build the matrix of the centers
        save_centers.append([cX,cY])

        # build the matrix of the particles
        for i,j in zip(part_x,part_y):
			save_paticles.append([i,j])

 
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)
    #cv2.imshow('thresh',thresh)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
