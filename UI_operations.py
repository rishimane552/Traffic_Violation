#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np

class DrawLineWidget(object):
    def __init__(self,img,title):
        self.read=False
        self.original_image =img
        self.clone = self.original_image.copy()
        self.title=title
        cv2.namedWindow(self.title)
        cv2.setMouseCallback(self.title, self.extract_coordinates)
        self.read=False
        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.image_coordinates = [np.array((x,y))]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            
            self.image_coordinates.append(np.array((x,y)))
            # print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            self.read=True
            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow(self.title, self.clone) 
            
        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

def select_ROI(image):
    roi = cv2.selectROI(image)
    image = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv2.destroyAllWindows()
    return image,roi

def line_widget(image,line_count,title):
    lines={}
    line_no=1
    draw_line_widget = DrawLineWidget(image,title)
    while True:
        cv2.imshow(title, draw_line_widget.show_image())
        key = cv2.waitKey(1)
        if draw_line_widget.read==True:
            draw_line_widget.read=False
            lines[line_no]={}
            lines[line_no]['Start']=draw_line_widget.image_coordinates[0]
            lines[line_no]['End']=draw_line_widget.image_coordinates[1]
            line_no+=1
        # Close program with keyboard 'q'
        
        if key == ord('q') or line_no==line_count+1:
            cv2.destroyAllWindows()
            break
    return lines

