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

save_particles = []
save_detections = []
colors = {"red" : (255, 0, 0), "green" : (0, 255, 0), "white" : (255, 255, 255), 
          "blue" : (0, 0, 255), "yellow" : (255, 255, 0) , "turquoise" : (0, 255, 255), "purple" : (255, 0, 255)}
number_particles = 100

#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/people-walking.mp4"
video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/mot1.mp4"

# minimum size of the contours that will be considered. It permits to not deal with very little detections (noise)
min_area = 100

nb = 0
val = 1000

# first frame in the video
firstFrame = None

# dictionary of unique detections
uniq_detection = dict()

threshold_compare_hist = 0.8
threshold_compare_dist = 100

cap = cv2.VideoCapture(video)
fgbg = cv2.BackgroundSubtractorMOG2()
#fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    #resize because of the performance
    width = 500
    frame = imutils.resize(frame, width=width)

    #learning rate set to 0
    #fgmask = fgbg.apply(frame)
    fgmask = fgbg.apply(frame, learningRate=1.0/10)

    # Dilation
    dilation = cv2.dilate(fgmask,None,iterations = 2);
    
    (cnts, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        #cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Compute the descriptor : histogram of the color of the detection
        # first convert the detection into a RGB image : OpenCV stores images in BGR format rather than RGB ; 
        # matplotlib is going to be used to display the results and matplotlib assumes the image is in RGB format
        # second the histogram is computed and then it is normalized
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame_rgb[x:x+w, y:y+h]], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()

        # if it is the first frame, add all the detections in the dictionary with their hist
        # test for this new data model
        if firstFrame is None:
            detect_group = len(uniq_detection)
            uniq_detection[len(uniq_detection)] = [hist,cX,cY,x,y,x+w,y+h,detect_group]
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
                uniq_detection[candidate_key[index]] = [hist,cX,cY,x,y,x+w,y+h,detect_group]
                cv2.putText(frame, str(candidate_key[index]), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # we must add a new detector
            else:
                detect_group = len(uniq_detection)
                uniq_detection[len(uniq_detection)] = [hist,cX,cY,x,y,x+w,y+h,detect_group]
                cv2.putText(frame, str(len(uniq_detection)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # print(len(uniq_detection))

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

        # Build the matrix of the detections respecting data model
        # Detection : histogram of colors, x_center, y_center, x_min_contour, y_min_contour, x_max_contour, 
        # y_max_contour, group of detection
        save_detections.append([hist,cX,cY,x,y,x+w,y+h,detect_group])

        # build the matrix of the particles respecting data model
        # particle : x_ord, y_ord, weight, x_detection_center, y_detection_center, frame_count_since_born,
        # initial motion direction, initial velocity
        weight = 0
        frame_born = 0
        for i,j in zip(part_x,part_y):
            # Initialisation of the motion direction : orthogonal to the closest image borders
            dist_right = width - i
            dist_up = j
            dist_left = i
            dist_bottom = width - j
            distance_border = [dist_right,dist_up,dist_left,dist_bottom]
            distance_min_border = min(distance_border)
            if distance_min_border == dist_right or distance_min_border == dist_left:
                init_motion_dir = [0,1]
            if distance_min_border == dist_up or distance_min_border == dist_bottom:
                init_motion_dir = [1,0]
            save_particles.append([i,j,weight,None,None,frame_born,init_motion_dir,[0,0]])


        # Print the data of a special frame
        # nb = nb + 1
        # if nb == 20:
        #     print("distance", dist_centers)

    ### Update the particles ###

    # Calculate the distance between each detection,particle pair.
    # for detect in save_detections:
    #     for particl in save_particles:
    #         d = [detect[0],detect[1]]
    #         p = [particl[0],particl[1]]
            #norm_d_p = ssp.distance.euclidean(d,p)

    # not anymore the first frame
    firstFrame = 1

 
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)
    #cv2.imshow('thresh',thresh)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
