# /Users/louisvilpoux/anaconda/bin/jupyter_mac.command ; exit;
# cd Documents/Manchester/Dissertation/trackingbydetection/

# repere used : x_ord go to the right (->) ; y_ord go to the bottom (|)

import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.spatial as ssp
import itertools
import datetime
import math

save_particles = []
save_detections = []
save_association = dict()

number_particles = 100

#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/mot1.mp4"
#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/pets4.mp4"
video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/highway31.mp4"

# Minimum size of the contours that will be considered. It permits to not deal with very little detections (noise)
min_area = 700


# First frame in the video
firstFrame = None

frame_number = 0

# Parameters in the formulas
alpha = 2

# Dictionary of unique detections
uniq_detection = dict()

threshold_compare_hist = 0.8

cap = cv2.VideoCapture(video)
fgbg = cv2.BackgroundSubtractorMOG2()
#fgbg = cv2.BackgroundSubtractorMOG()

# Start of the timestamp
ts = datetime.datetime.now()
total = 0

while(1):
    frame_number = frame_number + 1
    ts2 = datetime.datetime.now()
    ret, frame = cap.read()
    #resize because of the performance
    width = 500
    frame = imutils.resize(frame, width=width)
    #cv2.rectangle(frame, (0, 0), (500, 370), (0, 255, 255), 2)

    #learning rate set to 0
    #fgmask = fgbg.apply(frame)
    fgmask = fgbg.apply(frame, learningRate=0.1)

    # Dilation
    dilation = cv2.dilate(fgmask,None,iterations = 2);
    
    (cnts, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = 0

    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
 
        valid = valid + 1

		# compute the bounding box for the contour, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # compute the center of the contour for each detection. cX and cY are the coords of the detection center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 2, (255, 255, 255), -1)
        #cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Compute the descriptor : histogram of the color of the detection
        # first convert the detection into a RGB image : OpenCV stores images in BGR format rather than RGB ; 
        # matplotlib is going to be used to display the results and matplotlib assumes the image is in RGB format
        # second the histogram is computed and then it is normalized
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame_rgb[x:x+w, y:y+h]], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()

        # if it is the first frame, add all the detections in the dictionary with their histogram
        if firstFrame is None:
            detect_group = len(uniq_detection)
            velocity_target = 0
            timestamp = datetime.datetime.now()
            uniq_detection[len(uniq_detection)] = [hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h]
            cv2.putText(frame, str(len(uniq_detection)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Comparison of the descriptor of the past detections.
        # Try first if it is not the first frame and if we already have detect some objects.
        # Different methods to compare histograms :
        # cv2.cv.CV_COMP_CORREL ; cv2.cv.CV_COMP_CHISQR ; cv2.cv.CV_COMP_INTERSECT ; cv2.cv.CV_COMP_BHATTACHARYYA
        # Given a threshold for the comparison, we decide if it is a new detection or not
        # If it is not a new detection, update the values of the dictionary using the distance to the center.
        # If the value of the comparison is greater than a threshold, the histogram is considered as a candidate 
        # histogram. The distance between the centers will help us to understand from which detection does the 
        # candidate histogram belong. The candidate detections are those which have a similar histogram.
        # The best candidate is the detection that has its center closest to the testing detection center.
        # In the case that there is no candidate after this test, the histogram will be add as a new detection.
        # Different distances can be used :
        # sqeuclidean ; cosine ; correlation ; hamming ; jaccard
        # A text describes the number of the detection
        candidate_key = []
        candidate_dist = []
        if firstFrame is not None:
            used = False
            if len(uniq_detection) != 0:
                for key_detec, val_detec in uniq_detection.iteritems():
                    if cv2.compareHist(hist,val_detec[0],cv2.cv.CV_COMP_CORREL) > threshold_compare_hist:
                        dist_centers = ssp.distance.cdist([(val_detec[1],val_detec[2])],[(cX,cY)],'euclidean')[0][0]
                        used = True
                        candidate_key.append(key_detec)
                        candidate_dist.append(dist_centers)
            # we must update a detector
            if used == True:
                index = candidate_dist.index(min(candidate_dist))
                detect_group = candidate_key[index]
                timestamp = datetime.datetime.now() - ts
                timestamp = timestamp.total_seconds()
                velocity_target = min(candidate_dist) / timestamp
                uniq_detection[candidate_key[index]] = [hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h]
                cv2.putText(frame, str(candidate_key[index]), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # we must add a new detector
            else:
                detect_group = len(uniq_detection)
                velocity_target = 0
                timestamp = datetime.datetime.now()
                uniq_detection[len(uniq_detection)] = [hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h]
                cv2.putText(frame, str(len(uniq_detection)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # print(len(uniq_detection))

        # Build the matrix of the detections respecting data model
        # Detection : histogram of colors, x_center, y_center, x_min_contour, y_min_contour, x_max_contour, 
        # y_max_contour, group of detection, velocity_target, timestamp, size_target, index
        save_detections.append([hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h,len(save_detections)])


        # draw the particles
        # find the optimal size limit for the particles spread
        mean = [cX, cY]
        val_cov = min(w,h)
        cov = [[val_cov, 0], [0, val_cov]]
        part_x, part_y = np.random.multivariate_normal(mean, cov, number_particles).T

        # build the matrix of the particles respecting data model
        # particle : x_ord, y_ord, weight, x_detection_center, y_detection_center, size_target, frame_count_since_born,
        # initial motion direction, initial velocity
        weight = 0
        frame_born = 0
        particles = []
        for i,j in zip(part_x,part_y):
            # plot the particles
            #cv2.circle(frame,(int(i),int(j)),1,(0, 0, 255), 0)
            # Initialisation of the motion direction : orthogonal to the closest image borders
            dist_right = width - i
            dist_up = j
            dist_left = i
            dist_bottom = width - j
            distance_border = [dist_right,dist_up,dist_left,dist_bottom]
            distance_min_border = min(distance_border)
            if distance_min_border == dist_right:
                init_motion_dir = [-1,0]
            if distance_min_border == dist_left:
                init_motion_dir = [1,0]
            if distance_min_border == dist_up:
                init_motion_dir = [0,1]
            if distance_min_border == dist_bottom:
                init_motion_dir = [0,-1]
            particles.append([[i,j],weight,None,None,None,frame_born,init_motion_dir,[0,0]])
        # The particles are added to save_particles by tracker
        save_particles.append(particles)
        save_association[save_particles.index(particles)] = []



    total = total + valid
    #if frame_number == 105:
    if frame_number == 110:
       print total

    # not anymore the first frame
    firstFrame = 1

    #filename = "Results/Detection/threshold/{}/pets/pets_{}.jpg".format(threshold_compare_hist,frame_number)
    #filename = "Results/Detection/threshold/{}/highway/highway_{}.jpg".format(threshold_compare_hist,frame_number)
    #cv2.imwrite(filename, frame)

    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)


    # timestamp_fin = datetime.datetime.now() - ts
    # timestamp_fin = timestamp_fin.total_seconds()

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
