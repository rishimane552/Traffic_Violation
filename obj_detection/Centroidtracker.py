from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from datetime import datetime
#LIne related calulations done using this library
from FastLine import Line
from point import directionOfPoint


class CentroidTracker():
    def __init__(self,line, lane, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.data={}
        self.flag={}
        self.frame=None
        self.IN=line[1]
        self.OUT=line[2]
        self.in_line= Line(p1=self.IN['Start'],p2=self.IN['End'])
        self.out_line= Line(p1=self.OUT['Start'],p2=self.OUT['End'])
        self.disappeared_data=[]
        self.lane=lane
        self.centroid={}
        self.lane_data={}
        self.violation=[]
        self.obj_image={}
        
        # store the numbe,r of maximum consecutive frames a given object is allowed to be marked as "disappeared" until we need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid,obj_rect):
        # when registering an object we use the next available object
        # ID to store the centroid
        now=datetime.now()
        
        if self.nextObjectID not in self.data.keys():
            self.data[self.nextObjectID]=[]
            self.lane_data[self.nextObjectID]={'change':False, 'Lane':None}
        # else: 
        #     self.data[self.nextObjectID].append(['REREGISTETED',now.strftime("%H:%M:%S")])
        self.objects[self.nextObjectID] = centroid
        
        in_dist=self.in_line.distance_to(centroid)
        out_dist=self.out_line.distance_to(centroid)

        if in_dist < out_dist:
                self.data[self.nextObjectID].append(['IN',now.strftime("%H:%M:%S")])
            
        else:
                self.data[self.nextObjectID].append(['WRONG',now.strftime("%H:%M:%S")])                
                
        self.centroid[self.nextObjectID]=[]
        self.centroid[self.nextObjectID].append(centroid)
        x,y,w,h=obj_rect
        self.obj_image[self.nextObjectID]=[obj_rect,self.frame[y:y+h,x:x+w]]    
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, disappear):
        now=datetime.now()
        for objectID in disappear:
            centroid=self.objects[objectID]
            in_dist=self.in_line.distance_to(centroid)
            out_dist=self.out_line.distance_to(centroid)
            if out_dist<in_dist:
                self.data[objectID].append(['OUT',now.strftime("%H:%M:%S")])
            else: 
                self.data[objectID].append(['WRONG',now.strftime("%H:%M:%S")])
            
            
            # detect if any violation done
            if self.data[objectID][0][0]=='WRONG' and self.data[objectID][1][0]=='WRONG':
                    self.violation.append([objectID,'ONE WAY VIOLATION'])
            if self.lane_data[objectID]['change']==True:
                    self.violation.append([objectID,'LANE VIOLATION'])
                
            
            del self.objects[objectID]  
            del self.disappeared[objectID]


    def update(self, rects):
        if len(rects) == 0:
            ids = self.disappeared.keys()
            disappear=[]
            for objectID in ids:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    disappear.append(objectID)
            self.deregister(disappear)
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        obj_rect=[]
        for (i, (x , y , w , h)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
                x1 = int(w / 2)
                y1 = int(h / 2)
 
                cx = x + x1
                cy = y + y1
              
                inputCentroids[i] = (cx, cy)
                obj_rect.append([x,y,w,h])
                

        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i],obj_rect[i])
                

        # otherwise, are are currently tracking objects so we need to try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must 
            #(1) find the smallest value in each row and then 
            #(2) sort the row indexes based on their minimum values so that the row with the smallest value as at the *front* of the index list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by finding the smallest value in each column and then sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,or deregister an object we need to keep track of which of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row orcolumn value before, ignore it val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row, set its new centroid, and reset the disappeared counter
                objectID = objectIDs[row]
                
                self.objects[objectID] = np.array(inputCentroids[col])
                if self.obj_image[objectID]:
                    x,y,w,h=self.obj_image[objectID][0]
                    prev_area=w*h
                    x,y,w,h=obj_rect[col]
                    if w*h>prev_area:
                        self.obj_image[objectID]=[obj_rect[col],self.frame[y:y+h,x:x+w]]   
                else:
                    x,y,w,h=obj_rect[col]
                    self.obj_image[objectID]=[obj_rect[col],self.frame[y:y+h,x:x+w]]
                self.centroid[objectID].append(self.objects[objectID])

                position=directionOfPoint(self.lane[0]['Start'],self.lane[0]['End'],self.objects[objectID])

                if position==1:
                    if self.lane_data[objectID]['Lane']!=None:
                        if self.lane_data[objectID]['Lane']!='Right':
                            self.lane_data[objectID]['change']=True  
                    self.lane_data[objectID]['Lane']='Right'
                elif position==-1:
                    if self.lane_data[objectID]['Lane']!=None:
                        if self.lane_data[objectID]['Lane']!='Left':
                            self.lane_data[objectID]['change']=True  
                    self.lane_data[objectID]['Lane']='Left'
                elif position==0:
                    # self.lane_data[objectID]['Lane']='Centre'
                    pass
                self.disappeared[objectID] = 0
                

                # indicate that we have examined each of the row and column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is equal or greater than the number of input centroids
            # we need to check and see if some of these objects have potentially disappeared
            disappear=[]
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # if self.disappeared[objectID]==1:
                    #     top,bottom,left,right =get_crop_coord(self.frame,self.objects[objectID])
                    #     self.obj_image[objectID]=self.frame[top:bottom,left:right]

                    # check to see if the number of consecutive  frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        disappear.append(objectID)
                        
                self.deregister(disappear)

            # otherwise, if the number of input centroids is greater than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col],obj_rect[col])
                    
        

        return 
