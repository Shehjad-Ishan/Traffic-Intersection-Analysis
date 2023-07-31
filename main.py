import cv2
import torch
from tracker import *
import numpy as np
import math 
model = torch.hub.load('C://Users//USER//yolov5peoplecounterwin11', 'custom', path='projectweight.pt',force_reload=True, source='local', pretrained=False)


cap=cv2.VideoCapture('./drone.mp4')


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)
list=[]
tracking_objects={}
track_id=0

center_points_prev_frame=[]
count =0

area_1= [(122,368),(650,368),(688,375),(160,375)]
area1=set()
#Object detection and tracking
while True:
    center_points_cur_frame=[]
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame = cv2.resize(frame, (1020, 500))
    cv2.polylines(frame, [np.array(area_1,np.int32)],True,(0,255,0),1)
    results = model(frame)
    count+=1
    #frame = np.squeeze(result.render())
    
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        det_class = str(row['name'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate and draw midpoint
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center_points_cur_frame.append((cx,cy))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        
    if count <= 2:
        print(count)
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy= center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists=False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                
                #update object position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists= True
                    try:
                        center_points_cur_frame.remove(pt)
                    except:
                        print("NOT FOUND")
                    continue
            
            #Remove id
            if not object_exists:
                tracking_objects.pop(object_id)
                
        #Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id +=1
        
    
        
    
    
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0,0,255), -1)
        cv2.putText(frame, str(object_id), (pt[0],pt[1]-7), 0, 1, (0,0,255), 2)
        res=cv2.pointPolygonTest(np.array(area_1,np.int32),(int(pt[0]),int(pt[1])),False)
        
        if res>0:
            area1.add(object_id)
    p=len(area1)
    cv2.putText(frame,str(p),(20,30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    print("Tracking objects")
    print(tracking_objects)
                
                
    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)
    
    print ("PREV FRAME")
    print(center_points_prev_frame)
        
    # Display frame
    cv2.imshow('FRAME', frame)
    
    #Make a copy of the current frame
    center_points_prev_frame = center_points_cur_frame.copy()
    
    if cv2.waitKey(0) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()

    
    
