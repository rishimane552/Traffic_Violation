#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:11:55 2022

@author: aishwarya
"""

import cv2
import numpy as np
import UI_operations


#cap = cv2.VideoCapture('Datasets/Data1.mp4')

prototext_path='/home/aishwarya/traffic_violation/traffic_violation_api/models/MobileNetSSD_deploy.prototxt.txt'
model_path='/home/aishwarya/traffic_violation/traffic_violation_api/models/MobileNetSSD_deploy.caffemodel'
min_confidence=0.2

# List of categories and classes
categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
                4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
                9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
              13: 'horse', 14: 'motorbike', 15: 'person', 
              16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
              19: 'train', 20: 'tvmonitor'}
 
classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", 
            "diningtable",  "dog", "horse", "motorbike", "person", 
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# np.random.seed(543210)
# colors = np.random.uniform(255, 0, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)
class_selection=[2,6,7,14,15]


def nn_ssd_detection_model(frame):
    height,width=frame.shape[0],frame.shape[1]
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    IMG_NORM_RATIO = 0.007
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)),IMG_NORM_RATIO, (300,300), 130)
    
    # Set the input for the neural network
    net.setInput(blob)
     
    # Predict the objects in the image
    detected_objects = net.forward()
    rect=[]
    for i in range(detected_objects.shape[2]):
        class_index=int(detected_objects[0,0,i,1])
        confidence=detected_objects[0,0,i,2]
        if class_index in class_selection and confidence>=min_confidence:            
            upper_left_x= int(detected_objects[0,0,i,3]*width)
            upper_left_y= int(detected_objects[0,0,i,4]*height)
            lower_right_x=int(detected_objects[0,0,i,5]*width)
            lower_right_y=int(detected_objects[0,0,i,6]*height)
            h=lower_right_y-upper_left_y
            w=lower_right_x-upper_left_x
            rect.append([upper_left_x,upper_left_y,w,h])           
    return rect
        
#         # predict_text=categories[class_index]
#         cv2.rectangle(frame, (upper_left_x,upper_left_y), (lower_right_x,lower_right_y), colors[class_index],3)
#         cv2.putText(frame, predict_text, 
#                     (upper_left_x,upper_left_y-15 if upper_left_y>30 else upper_left_y+15), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[class_index],2)
        
#     cv2.imshow('Detected', frame)
#     cv2.waitKey(3)
# cv2.destroyAllWindows()
# cap.release()

