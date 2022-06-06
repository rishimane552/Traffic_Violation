#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:48:12 2022

@author: aishwarya
"""

class point:
      
    def __init__(self):
          
        self.x = 0
        self.y = 0
   
# Constant integers for directions
RIGHT = 1
LEFT = -1
ZERO = 0
   
def directionOfPoint(start, end, pt):
      
    global RIGHT, LEFT, ZERO
    
    
    A = point()#start
    B = point()#end
    P = point()#centroid
      
    A.x = start[0]
    A.y = start[1]
    B.x = end[0]
    B.y = end[1]
    P.x = pt[0]
    P.y = pt[1] # P(15, 28)
    # Subtracting co-ordinates of 
    # point A from B and P, to 
    # make A as origin
    B.x -= A.x
    B.y -= A.y
    P.x -= A.x
    P.y -= A.y
   
    # Determining cross Product
    cross_product = B.x * P.y - B.y * P.x
   
    # Return RIGHT if cross product is positive
    if (cross_product > 0):
        return RIGHT
          
    # Return LEFT if cross product is negative
    if (cross_product < 0):
        return LEFT
   
    # Return ZERO if cross product is zero
    return ZERO
  
