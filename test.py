import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.spatial as ssp
import scipy.stats as ss
import itertools
import datetime
import math
import random

save_particles = []
save_detections = []
dict_detection = dict()
dict_particle = dict()
colors = {"red" : (255, 0, 0), "green" : (0, 255, 0), "white" : (255, 255, 255), 
          "blue" : (0, 0, 255), "yellow" : (255, 255, 0) , "turquoise" : (0, 255, 255), "purple" : (255, 0, 255)}

number_particles = 50

video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/mot1.mp4"
#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/pets.mp4"
#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/highway.mp4"

# Minimum size of the contours that will be considered. It permits to not deal with very little detections (noise)
min_area = 300
#min_area = 700
#min_area = 500

nb = 0

# First frame in the video
firstFrame = None

frame_number = 0
frame_number_delete = 5

# Parameters in the formulas
alpha = 1
beta = 10
gamma = 1
etha = 0.5

threshold_compare_hist_dist = 0.8

# TO DEFINE
threshold_velocity_target = 0

# TO DEFINE
threshold_matching_score = 0

# TO DEFINE
nb_detect_save = 5

cap = cv2.VideoCapture(video)
fgbg = cv2.BackgroundSubtractorMOG2()
#fgbg = cv2.BackgroundSubtractorMOG()

# Normal distribution
standard_dev = 0.2
normal_distrib = ss.norm(0,standard_dev)

# Start of the timestamp
ts = datetime.datetime.now()


while(1):
    save_association = dict()
    frame_number = frame_number + 1
    ts2 = datetime.datetime.now()
    ret, frame = cap.read()
    # Resize because of the performance
    width = 500
    frame = imutils.resize(frame, width=width)

    #learning rate set to 0
    #fgmask = fgbg.apply(frame)
    fgmask = fgbg.apply(frame, learningRate=1.0/10)

    # Dilation
    dilation = cv2.dilate(fgmask,None,iterations = 2);
    
    (cnts, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Limit of the zone for the instantiation of the new trackers
    x_limit = width/8
    y_limit = width/8
    x_w_limit = 9*width/10
    y_h_limit = 7*width/10
    cv2.rectangle(frame, (x_limit, y_limit), (x_w_limit, y_h_limit), (0, 255, 255), 2)

    index_add = 0
    to_add_to_dict_particle = dict()

    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
 
        # Compute the bounding box for the contour, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        color_detect = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        # Compute the center of the contour for each detection. cX and cY are the coords of the detection center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 2, (255, 255, 255), -1)

        # Compute the descriptor : histogram of the color of the detection
        # First convert the detection into a RGB image : OpenCV stores images in BGR format rather than RGB ; 
        # matplotlib is going to be used to display the results and matplotlib assumes the image is in RGB format
        # Second the histogram is computed and then it is normalized
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame_rgb[x:x+w, y:y+h]], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()

        # If it is the first frame, add all the detections in the dictionary with their histogram
        if firstFrame is None:
            detect_group = len(dict_detection)
            velocity_target = 0
            timestamp = datetime.datetime.now()
            dict_detection[len(dict_detection)] = []
            dict_detection[len(dict_detection)-1].append([hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h,color_detect])
            cv2.putText(frame, str(len(dict_detection)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_detect, 2)

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
            if len(dict_detection) != 0:
                for key_detec, val_detec in dict_detection.iteritems():
                    if len(val_detec) != 0:
                        comp_hist_dist = cv2.compareHist(hist,val_detec[len(val_detec)-1][0],cv2.cv.CV_COMP_CORREL)
                        if comp_hist_dist > threshold_compare_hist_dist:
                            dist_centers = ssp.distance.cdist([(val_detec[len(val_detec)-1][1],val_detec[len(val_detec)-1][2])],[(cX,cY)],'euclidean')[0][0]
                            used = True
                            candidate_key.append(key_detec)
                            candidate_dist.append(dist_centers)
            # We must update a detector
            # Build the matrix of the detections respecting data model
            # Detection : histogram of colors, x_center, y_center, x_min_contour, y_min_contour, x_max_contour, 
            # y_max_contour, group of detection, velocity_target, timestamp, size_target, index
            if used == True:
                index = candidate_dist.index(min(candidate_dist))
                detect_group = candidate_key[index]
                timestamp = datetime.datetime.now() - ts
                timestamp = timestamp.total_seconds()
                velocity_target = min(candidate_dist) / timestamp
                color_group = dict_detection[candidate_key[index]][len(dict_detection[candidate_key[index]]) - 1][11]
                dict_detection[candidate_key[index]].append([hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h,color_group])
                cv2.putText(frame, str(candidate_key[index]), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_group, 2)
            # We must add a new detector
            else:
                detect_group = len(dict_detection)
                velocity_target = 0
                timestamp = datetime.datetime.now()
                dict_detection[len(dict_detection)] = []
                dict_detection[len(dict_detection)-1].append([hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h,color_detect])
                cv2.putText(frame, str(len(dict_detection)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_detect, 2)


        # draw the particles
        # find the optimal size limit for the particles spread
        mean = [cX, cY]
        val_cov = min(w,h)
        cov = [[val_cov, 0], [0, val_cov]]
        part_x, part_y = np.random.multivariate_normal(mean, cov, number_particles).T

        # Build the matrix of the particles respecting data model
        # particle : [x_ord, y_ord], weight, x_detection_center, y_detection_center, size_target, frame_count_since_born,
        # initial motion direction, initial velocity, future key in the dict
        weight = 1
        success_track_frame = 1
        frame_born = frame_number
        particles = []
        for i,j in zip(part_x,part_y):
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
            # if (i < x_limit or i > x_w_limit or j < y_limit or j > y_h_limit) and frame_number > 2:
                # particles.append([[i,j],weight,None,None,None,frame_born,init_motion_dir,[0,0],len(dict_particle),success_track_frame])
	            # Plot the particles
                #cv2.circle(frame,(int(i),int(j)),1,(0, 0, 255), 0)
            # if frame_number <= 2:
            particles.append([[i,j],weight,None,None,None,frame_born,init_motion_dir,[0,0],len(dict_particle),success_track_frame,(255,255,255)])
	            # Plot the particles
                #cv2.circle(frame,(int(i),int(j)),1,(0, 0, 255), 0)            	
        # The particles are added to save_particles by tracker
        #dict_particle[len(dict_particle)] = particles
        if particles != []:
            to_add_to_dict_particle[len(dict_particle)+index_add] = particles

        index_add = index_add + 1

    print "idx", index_add
    print "avant instan",len(dict_particle)
   ## Instantiation of the new trackers ##

    for key_track, track in to_add_to_dict_particle.iteritems():
        # if dict_particle.has_key(key_track):
        #     print len(dict_particle),key_track
        dict_particle[key_track] = track

    ## End of the Instantiation of the new Trackers ##



    print "apres instan",len(dict_particle)
    ## Print the Particles ##

    for part in list(itertools.chain.from_iterable(dict_particle.values())):
        cv2.circle(frame,(int(part[0][0]),int(part[0][1])),3,(0, 0, 255), 0)

    # for part in list(itertools.chain.from_iterable(to_add_to_dict_particle.values())):
    #     cv2.circle(frame,(int(part[0][0]),int(part[0][1])),3,part[10], 0)  

    ## End of the Print of the Particles ##

    print "########"








    # not anymore the first frame
    firstFrame = 1

    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break



cap.release()
cv2.destroyAllWindows()
