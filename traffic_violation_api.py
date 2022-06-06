from traffic_violation import init_frame_param,violation_detection
from Centroidtracker import CentroidTracker
import requests
import sys
import cv2
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle


api = Flask(__name__)
CORS(api)

tracker=''
Intialized_data=''
frame_no=0

@api.route('/Initialisation', methods=['POST', 'GET'])
def Initialisation():
    global tracker
    global Intialized_data
    global frame_no

    frame_no=0
    data = request.json
    print("Entered Init")
    print(data)
    print(f'tracker --{tracker}')
    try:
        UI_data_Init=init_frame_param()
        UI_data_Init.TowardsCamera=data["TowardsCamera"]
        UI_data_Init.crop_coord=[data["ImageProperties"]["x"],data["ImageProperties"]["y"],data["ImageProperties"]["height"],data["ImageProperties"]["width"]]
        UI_data_Init.boundary_lines={}
        UI_data_Init.boundary_lines[1]=data["LineProperties"][0]
        UI_data_Init.boundary_lines[2]=data["LineProperties"][1]
        print(UI_data_Init.boundary_lines)
        UI_data_Init.lane_lines=data["LaneProperties"]
        print(UI_data_Init)
        UI_data_Init.rearrange_line()
        params = {'camera_flag': UI_data_Init.TowardsCamera, 'crop_coord' : UI_data_Init.crop_coord,'boundary_lines': UI_data_Init.boundary_lines, 'lane_lines': UI_data_Init.lane_lines}
        print(params)
        Intialized_data=UI_data_Init
        #with open('params.json', 'w') as f:
        #    json.dump(params, f)
        print ('Initializing tracker')
        tracker = CentroidTracker(UI_data_Init.boundary_lines, UI_data_Init.lane_lines)
        print(f'tracker initialised')
        return jsonify({'flag': True, 'data':data})
    except Exception as e :
        return jsonify({'flag': False, 'data':f"Initialisation failed {e}"})


@api.route('/traffic_violation', methods=['POST', 'GET'])
def traffic_violation():
    global tracker
    global Intialized_data
    global frame_no

    frame_no+=1
    print("=============================")
    print(frame_no)
    print("=============================")

    data = request.json
    start_time = time.time()
    #print(start_time)
    blob = data["image"] 
    #with open('params.json', 'r') as f:
    #        params = json.loads(f.read())
    print(f'tracker before update {tracker}')
    print(f'initialised data -{Intialized_data}')
    violated = violation_detection(tracker,Intialized_data,blob)
    tracker=violated.tracker
    #if violated.result!=[]:
    print ("Violations")
    print(violated.result)
    #print(f'tracker updated {tracker}')
    return jsonify({'flag': True, 'Violations':violated.result})


if __name__ == '__main__':
    api.run(host='0.0.0.0',port=5001)
    #traffic_violation()
    CORS(api)





