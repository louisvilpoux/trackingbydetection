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

number_particles = 100

#video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/people-walking.mp4"
video = "/Users/louisvilpoux/Documents/Manchester/Dissertation/Data/mot1.mp4"

# Minimum size of the contours that will be considered. It permits to not deal with very little detections (noise)
min_area = 100

nb = 0

# First frame in the video
firstFrame = None

frame_number = 0
frame_number_delete = 5

# Parameters in the formulas
alpha = 2
beta = 20
gamma = 2
etha = 1

threshold_compare_hist = 0.85

# TO DEFINE
threshold_velocity_target = 1000

# TO DEFINE
threshold_matching_score = 0

cap = cv2.VideoCapture(video)
fgbg = cv2.BackgroundSubtractorMOG2()
#fgbg = cv2.BackgroundSubtractorMOG()

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

    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
 
        # Compute the bounding box for the contour, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
            dict_detection[len(dict_detection)-1].append([hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h])
            cv2.putText(frame, str(len(dict_detection)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
                        if cv2.compareHist(hist,val_detec[len(val_detec)-1][0],cv2.cv.CV_COMP_CORREL) > threshold_compare_hist:
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
                dict_detection[candidate_key[index]].append([hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h])
                cv2.putText(frame, str(candidate_key[index]), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # We must add a new detector
            else:
                detect_group = len(dict_detection)
                velocity_target = 0
                timestamp = datetime.datetime.now()
                dict_detection[len(dict_detection)] = []
                dict_detection[len(dict_detection)-1].append([hist,cX,cY,x,y,x+w,y+h,detect_group,velocity_target,timestamp,w*h])
                cv2.putText(frame, str(len(dict_detection)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


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
        to_add_to_dict_particle = dict()
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
            if (i < x_limit or i > x_w_limit or j < y_limit or j > y_h_limit) and frame_number > 2:
                particles.append([[i,j],weight,None,None,None,frame_born,init_motion_dir,[0,0],len(dict_particle),success_track_frame])
	            # Plot the particles
                cv2.circle(frame,(int(i),int(j)),1,(0, 0, 255), 0)
            if frame_number <= 2:
                particles.append([[i,j],weight,None,None,None,frame_born,init_motion_dir,[0,0],len(dict_particle),success_track_frame])
	            # Plot the particles
                cv2.circle(frame,(int(i),int(j)),1,(0, 0, 255), 0)            	
        # The particles are added to save_particles by tracker
        #dict_particle[len(dict_particle)] = particles
        if particles != []:
        	to_add_to_dict_particle[len(dict_particle)] = particles



    ### Delete not associated Particles ###

    for part in list(itertools.chain.from_iterable(dict_particle.values())):
        if frame_number - part[5] > frame_number_delete and part[2] == None:
            idx = part[8]
            dict_particle[idx].remove(part)
            if len(dict_particle[idx]) == 0:
                dict_particle.pop(idx,None)

    ### End of Delete not associated Particles ###



    ### Data Association ###

    # Calculate the distance between each detection,tracker pair.
    # Because two for loops is too computationaly expensive, another way to try all the possible values of both lists
    # has been found. It used the Python library itertools and make the product of the data of both lists.
    # From all the associations available, pick the assocation that has the best score (greater than the threshold)
    #if save_detections != []:
    if dict_detection != dict():
        # The loop iterate over the trackers and not the particles
        prev_detect = None
        standard_dev = 0.5
        normal_distrib1 = ss.norm(0,standard_dev)
        for tracker,detect in list(itertools.product(dict_particle.values(),list(itertools.chain.from_iterable(dict_detection.values())))):
            d = [detect[1],detect[2]]
            size_detection = detect[10]
            group = detect[7]
            if prev_detect != None and prev_detect[7] == group:
                size_tracker = prev_detect[10]
                pos_tracker = [prev_detect[1],prev_detect[2]]
                velocity = detect[8]
                agreement_target_detection = normal_distrib1.cdf((size_tracker - size_detection) / float(size_tracker))
                if abs(velocity) < threshold_velocity_target and abs(ssp.distance.euclidean(d,pos_tracker)) != 0:
                    gating = agreement_target_detection * normal_distrib1.cdf(abs(ssp.distance.euclidean(d,pos_tracker)))
                else:
                    distance_detection_motiontracker = (abs(tracker[0][6][0]*detect[1] + tracker[0][6][1]*detect[2]))/math.sqrt((tracker[0][6][0]**2)+(tracker[0][6][1]**2))
                    gating = agreement_target_detection * normal_distrib1.cdf(distance_detection_motiontracker)
                sum_part_tracker = 0
                for part in tracker:
                    sum_part_tracker = sum_part_tracker + normal_distrib1.cdf(ssp.distance.euclidean(d,part[0]))
                matching_score = gating * (1 + alpha * sum_part_tracker)
                if matching_score > threshold_matching_score:
                    index_tracker = tracker[0][8]
                    if save_association.has_key(index_tracker):
                        if matching_score > save_association[index_tracker][0][2]:
                            save_association[index_tracker] = []
                            save_association[index_tracker].append([tracker,detect,matching_score])
                        else:
							continue
                    else:
                        save_association[index_tracker] = []
                        save_association[index_tracker].append([tracker,detect,matching_score])
            prev_detect = detect

    ### End of Data Association ###



    ### Bootstrap Filter : Observation Model ###
    # Choose this parameter
    standard_dev = 0.5
    normal_distrib2 = ss.norm(0,standard_dev)
    for partic in list(itertools.chain.from_iterable(dict_particle.values())):
        key = partic[8]
        if save_association.has_key(key):
            coord_part = partic[0]
            coord_detec = [save_association[key][0][1][1],save_association[key][0][1][2]]
            # Detection term
            detection_term = beta * 1 * normal_distrib2.cdf(abs(ssp.distance.euclidean(coord_part,coord_detec)))
            # Update the data obtained from the association
            partic[2] = save_association[key][0][1][1]
            partic[3] = save_association[key][0][1][2]
            partic[4] = save_association[key][0][1][10]
            partic[9] = partic[9] + 1
            # Classifier term
            classifier_term = 1
            new_weight = detection_term + classifier_term
            partic[1] = (1/float(number_particles)) * new_weight

    ### End of the Bootstrap Filter : Observation Model ###



    ### Resampling ###

    copy_dict_particle = dict()
    for key,track in dict_particle.iteritems():
        sum_part = 0
        norm_weight = []
        weight = []
        # Sum of the weights
        for part in track:
            sum_part = sum_part + part[1]
        # Normalize the weight
        for part in track:
            norm_weight.append(part[1]/float(sum_part))
            weight.append(track.index(part))
        random_weight = np.random.choice(weight,number_particles,p=norm_weight)
        # Build the new sample of particles
        copy_dict_particle[key] = []
        for idx in random_weight:
            copy_dict_particle[key].append(track[idx])
    dict_particle = dict()
    dict_particle = copy_dict_particle

    ### End of Resampling ###



    ### Propagation ###

    copy_dict_particle = dict()
    for part in list(itertools.chain.from_iterable(dict_particle.values())):
        old_position = part[0]
        old_velocity = part[7]
        key = part[8]
        if part[4] != None:
            noise_position = np.random.normal(0,part[4])
        else:
            noise_position = 0
        noise_velocity = np.random.normal(0,1/float(part[9]))
        timestamp = datetime.datetime.now() - ts2
        timestamp = timestamp.total_seconds()
        new_position = [old_position[0] + old_velocity[0] * timestamp + noise_position , old_position[1] + old_velocity[1] * timestamp + noise_position]
        new_velocity = [old_velocity[0] + noise_velocity , old_velocity[1] + noise_velocity]
        new_part = [new_position,part[1],part[2],part[3],part[4],part[5],part[6],new_velocity,key,part[9]]
        if new_position[0] > 0 or new_position[0] < width or new_position[1] > 0 or new_position[1] < width:
	        if copy_dict_particle.has_key(key):
	            copy_dict_particle[key].append(new_part)
	        else:
	            copy_dict_particle[key] = []
	            copy_dict_particle[key].append(new_part)
    dict_particle = dict()
    dict_particle = copy_dict_particle

    ### End of the Propagation ###



   ## Instantiation of the new trackers ##

    for key_track, track in to_add_to_dict_particle.iteritems():
        dict_particle[key_track] = track

    ## End of the Instantiation of the new Trackers ##




    # Print the data of a special frame
    # nb = nb + 1
    # if nb == 10:
    #     print("save_association", save_association)


    # not anymore the first frame
    firstFrame = 1

    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame',frame)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break



cap.release()
cv2.destroyAllWindows()
