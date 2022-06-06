#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:17:17 2022

@author: rishikesh 
"""

import json
import cv2
import UI_operations
#from Centroidtracker import CentroidTracker
import numpy as np
from MobileNetSSD import nn_ssd_detection_model
import _pickle as cPickle
from PIL import Image
import base64
import io

#with open('params.json', 'r') as f:
#        data = json.loads(f.read())

class init_frame_param():
    def __init__(self):      
        self.boundary_lines=[]
        self.lane_lines=[]
        self.crop_coord=[]
        self.TowardsCamera=[]
    
    def rearrange_line(self):
        if self.TowardsCamera==False:
            for lane in self.lane_lines:
                if lane['Start'][1]<lane['End'][1]:
                    temp=lane['Start']
                    lane['Start']=lane['End']
                    lane['End']=temp
                    self.lane_lines[0]=lane
            if self.boundary_lines[1]['Start'][1]< self.boundary_lines[2]['Start'][1]:
                temp=self.boundary_lines[1]
                self.boundary_lines[1]=self.boundary_lines[2]
                self.boundary_lines[2]=temp 

        else:
            for lane in self.lane_lines:
                if lane['Start'][1]>lane['End'][1]:
                    temp=lane['Start']
                    lane['Start']=lane['End']
                    lane['End']=temp
                    self.lane_lines[0]=lane
                
            if self.boundary_lines[1]['Start'][1]> self.boundary_lines[2]['Start'][1]:
                temp=self.boundary_lines[1]
                self.boundary_lines[1]=self.boundary_lines[2]
                self.boundary_lines[2]=temp


        

class violation_detection():
    def __init__(self,tracker,intialized_data,blob):      
        self.intialized_data=intialized_data
        self.extract_ROI_frame(blob)
        #self.object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
        self.tracker = tracker
        self.processing()

    def extract_ROI_frame(self,blob):
        content = blob.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        image=cv2.cvtColor(np.array(Image.open(io.BytesIO(body))), cv2.COLOR_BGR2RGB)
        print ("blob is converted to image with image size of")
        print(image.shape[:2])
        x,y,w,h=self.intialized_data.crop_coord
        self.frame=image[x:x+w,y:y+h]
        print("ROI is cropped and ROI Image size is")
        print(self.frame.shape[:2])
        #kernel = np.array([[0, -1, 0],
        #           [-1, 5,-1],
        #           [0, -1, 0]])
        #self.frame = cv2.filter2D(src=self.frame, ddepth=-1, kernel=kernel)
        return

    def cluster_centroids(self,box):
        x,y,w,h = box
        flag=False
        for r in self.rects:
              x1 = int(w / 2)
              y1 = int(h / 2)
              cx = x + x1
              cy = y + y1
              
              x1 = int(r[2] / 2)
              y1 = int(r[3] / 2)
              rcx = r[0] + x1
              rcy = r[1] + y1
               
              if abs(cx-rcx)<40 and abs(cy-rcy)<60:
                   flag=False
                   break
              else:
                   flag=True   
        return flag    
    
    
    def processing(self):
        self.rects=[]
        # self.cascade_cars()
        # self.background_sub_method()
        self.nn_ssd_method()
        self.tracker.frame=self.frame
        self.tracker.update(self.rects)
        print("tracked objects")
        print(self.tracker.obj_image.keys())
        print(self.tracker.violation)
        self.result=[]
        if len(self.tracker.violation)>0:
            for violation in self.tracker.violation:
                img = self.tracker.obj_image[violation[0]][1]
                string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
                #string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
                self.result.append([string,violation[1]])
                #cv2.imwrite(violation[1]+str(violation[0])+'.jpg',tracker.obj_image[violation[0]][1])
            self.tracker.violation=[]
        # loop over the tracked objects
        #for (objectID, centroid) in self.tracker.objects.items():
        #    text = "ID {}".format(objectID)
        #    cv2.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10),
        #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #    cv2.circle(self.frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        #    cv2.imshow('frame', self.frame)
        return
    
    def nn_ssd_method(self):
        print("MobileSSD method detected following vechiclesi at ")
        vehicles=nn_ssd_detection_model(self.frame)
        print (vehicles)

        #self.orig_frame=copy.deepcopy(self.frame)
        for box in vehicles:
          x,y,w,h = box
          self.rects.append(box)
          #cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0))
        
        
    def cascade_cars(self):
        car_cascade = cv2.CascadeClassifier('Datasets/cars.xml')
        cars = car_cascade.detectMultiScale(self.frame, 4, 1)
        #self.orig_frame=copy.deepcopy(self.frame)
        for box in cars:
          x,y,w,h = box
          if len(self.rects)==0 or self.cluster_centroids(box)==True:
                self.rects.append(box)
                #cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0))
                
    def background_sub_method(self):
        mask=self.object_detector.apply(self.frame)
        _,mask=cv2.threshold(mask,254,255,cv2.THRESH_OTSU)  
        contours,_=cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
        #self.orig_frame=copy.deepcopy(self.frame)
        for cnt in contours:
          area=cv2.contourArea(cnt)
          if area>400:
              x,y,w,h = cv2.boundingRect(cnt)
              box = cv2.boundingRect(cnt)
              if len(self.rects)==0 or self.cluster_centroids(box)==True:
                    self.rects.append(box)
                    #cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0))
                    

        
    
#x=main()
