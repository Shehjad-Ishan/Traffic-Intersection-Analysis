import cv2
import torch
from tracker import *
import numpy as np
import math 
model = torch.hub.load('C://Users//USER//YOLOv5-Flask1', 'custom', path='projectweight.pt',force_reload=True, source='local')


cap=cv2.VideoCapture('./traf720.mp4')


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

#####Declare Variables#########
list=[]
tracking_objects={}
track_id=0

center_points_prev_frame=[]
count =0

area_nrt_in= [(308,282),(331,288),(340,243),(321,240)]
area_nrt_out =[(340,243),(321,240),(330,206),(350,207)]
area_est_in = [(413,136),(403,168),(528,151),(519,121)]
area_est_out =[(700,172),(726,154),(799,185),(778,211)]
area_sth_out = [(798,235),(748,299),(785,311),(827,252)]
area_sth_in= [(712,358),(748,299),(785,311),(744,366)]
area_wst_out =[(319,312),(467,354),(448,389),(310,351)]
area_wst_in =[(468,354),(617,399),(603,433),(448,390)]
#########################################
areaNin=set()
areaNout=set([1,2,3])
areaSout=set()
areaSin=set()
areaEin=set([1,2])
areaEout=set()
areaWin=set()
areaWout=set()
##########################################
n2n=0
n2s=0
n2e=0
n2w=0
s2s=0
s2n=0
s2e=0
s2w=0
e2e=0
e2s=0
e2w=0
e2n=0
w2w=0
w2e=0
w2n=0
w2s=0
#######################################
common_elements_Nout_Ein=set()
common_elements_Nout_Win=set()
common_elements_Nout_Sin=set()
common_elements_Nout_Nin=set()
common_elements_Sout_Nin=set()
common_elements_Sout_Win=set()
common_elements_Sout_Ein=set()
common_elements_Sout_Sin=set()
common_elements_Eout_Nin=set()
common_elements_Eout_Sin=set()
common_elements_Eout_Win=set()
common_elements_Eout_Ein=set()
common_elements_Wout_Nin=set()
common_elements_Wout_Sin=set()
common_elements_Wout_Ein=set()
common_elements_Wout_Win=set()
############################################
common_elements_Nout_Ein_prev = set()
common_elements_Nout_Win_prev = set()
common_elements_Nout_Sin_prev = set()
common_elements_Nout_Nin_prev = set()
common_elements_Sout_Nin_prev = set()
common_elements_Sout_Win_prev = set()
common_elements_Sout_Ein_prev = set()
common_elements_Sout_Sin_prev = set()
common_elements_Eout_Nin_prev = set()
common_elements_Eout_Sin_prev = set()
common_elements_Eout_Win_prev = set()
common_elements_Eout_Ein_prev = set()
common_elements_Wout_Nin_prev = set()
common_elements_Wout_Sin_prev = set()
common_elements_Wout_Ein_prev = set()
common_elements_Wout_Win_prev = set()




######Object detection and tracking
while True:
    center_points_cur_frame=[]
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame = cv2.resize(frame, (1020, 500))
    cv2.polylines(frame, [np.array(area_nrt_in,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame, [np.array(area_nrt_out,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame, [np.array(area_wst_out,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame, [np.array(area_wst_in,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame, [np.array(area_sth_in,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame, [np.array(area_sth_out,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame, [np.array(area_est_in,np.int32)],True,(0,255,0),1)
    cv2.polylines(frame, [np.array(area_est_out,np.int32)],True,(0,255,0),1)
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
        cv2.putText(frame, str(object_id), (pt[0],pt[1]-7), 0, 0.5, (0,0,255), 1)
        resNin=cv2.pointPolygonTest(np.array(area_nrt_in,np.int32),(int(pt[0]),int(pt[1])),False)
        resEin=cv2.pointPolygonTest(np.array(area_est_in,np.int32),(int(pt[0]),int(pt[1])),False)
        resWin=cv2.pointPolygonTest(np.array(area_wst_in,np.int32),(int(pt[0]),int(pt[1])),False)
        resSin=cv2.pointPolygonTest(np.array(area_sth_in,np.int32),(int(pt[0]),int(pt[1])),False)
        resNout=cv2.pointPolygonTest(np.array(area_nrt_out,np.int32),(int(pt[0]),int(pt[1])),False)
        resSout=cv2.pointPolygonTest(np.array(area_sth_out,np.int32),(int(pt[0]),int(pt[1])),False)
        resEout=cv2.pointPolygonTest(np.array(area_est_out,np.int32),(int(pt[0]),int(pt[1])),False)
        resWout=cv2.pointPolygonTest(np.array(area_wst_out,np.int32),(int(pt[0]),int(pt[1])),False)
        
        if resNin>0:
            areaNin.add(object_id)
        if resNout>0:
            areaNout.add(object_id)
        if resSin>0:
            areaSin.add(object_id)
        if resSout>0:
            areaSout.add(object_id)
        if resEin>0:
            areaEin.add(object_id)
        if resEout>0:
            areaEout.add(object_id)
        if resWin>0:
            areaWin.add(object_id)
        if resWout>0:
            areaWout.add(object_id)
            
    ##Check if the element is common in North out and S,W,E in boxes##      
            
    common_elements_Nout_Ein=areaNout.intersection(areaEin)
    common_elements_Nout_Win=areaNout.intersection(areaWin)
    common_elements_Nout_Sin=areaNout.intersection(areaSin)
    common_elements_Nout_Nin=areaNout.intersection(areaNin)
    
    
    
    
    if common_elements_Nout_Ein:
        common_elements_Nout_Ein_check= common_elements_Nout_Ein.intersection(common_elements_Nout_Ein_prev)
        n2e+=len(common_elements_Nout_Ein)- len(common_elements_Nout_Ein_check)
        common_elements_Nout_Ein_prev=common_elements_Nout_Ein.copy()
    if common_elements_Nout_Win:
        common_elements_Nout_Win_check= common_elements_Nout_Win.intersection(common_elements_Nout_Win_prev)
        n2w+=len(common_elements_Nout_Win)- len(common_elements_Nout_Win_check)
        common_elements_Nout_Win_prev= common_elements_Nout_Win.copy()
    if common_elements_Nout_Sin:
        common_elements_Nout_Sin_check= common_elements_Nout_Sin.intersection(common_elements_Nout_Sin_prev)
        n2s+=len(common_elements_Nout_Sin)-len(common_elements_Nout_Sin_check)
        common_elements_Nout_Sin_prev= common_elements_Nout_Sin.copy()
    if common_elements_Nout_Nin:
        common_elements_Nout_Nin_check= common_elements_Nout_Nin.intersection(common_elements_Nout_Nin_prev)
        n2n+=len(common_elements_Nout_Nin)-len(common_elements_Nout_Nin_check)
        common_elements_Nout_Nin_prev= common_elements_Nout_Nin.copy()
        
    ##Check if the element is common in South out and N,W,E in boxes##
        
    common_elements_Sout_Nin=areaSout.intersection(areaNin)
    common_elements_Sout_Win=areaSout.intersection(areaWin)
    common_elements_Sout_Ein=areaSout.intersection(areaEin)
    common_elements_Sout_Sin=areaSout.intersection(areaSin)
    
    if common_elements_Sout_Nin:
        common_elements_Sout_Nin_check= common_elements_Sout_Nin.intersection(common_elements_Sout_Nin_prev)
        s2n+=len(common_elements_Sout_Nin)- len(common_elements_Sout_Nin_check)
        common_elements_Sout_Nin_prev= common_elements_Sout_Nin.copy()
    if common_elements_Sout_Win:
        common_elements_Sout_Win_check= common_elements_Sout_Win.intersection(common_elements_Sout_Win_prev)
        s2w+=(common_elements_Sout_Win)-len(common_elements_Sout_Win_check)
        common_elements_Sout_Win_prev= common_elements_Sout_Win.copy()
    if common_elements_Sout_Ein:
        common_elements_Sout_Ein_check= common_elements_Sout_Ein.intersection(common_elements_Sout_Ein_prev)
        s2e+=len(common_elements_Sout_Ein)- len(common_elements_Sout_Ein_check)
        common_elements_Sout_Ein_prev= common_elements_Sout_Ein.copy()
    if common_elements_Sout_Sin:
        common_elements_Sout_Sin_check= common_elements_Sout_Sin.intersection(common_elements_Sout_Sin_prev)
        s2s+=len(common_elements_Sout_Sin)- len(common_elements_Sout_Sin_check)
        common_elements_Sout_Sin_prev= common_elements_Sout_Sin.copy()
        
    ##Check if the element is common in East out and S,W,N in boxes##
        
    common_elements_Eout_Nin=areaEout.intersection(areaNin)
    common_elements_Eout_Sin=areaEout.intersection(areaSin)
    common_elements_Eout_Win=areaEout.intersection(areaWin)
    common_elements_Eout_Ein=areaEout.intersection(areaEin)
    
    
    if common_elements_Eout_Nin:
        common_elements_Eout_Nin_check= common_elements_Eout_Nin.intesection(common_elements_Eout_Nin_prev)
        e2n+=len(common_elements_Eout_Nin)- len(common_elements_Eout_Nin_check)
        common_elements_Eout_Nin_prev= common_elements_Eout_Nin.copy()
    if common_elements_Eout_Win:
        common_elements_Eout_Win_check= common_elements_Eout_Win.intersection(common_elements_Eout_Win_prev)
        e2w+=len(common_elements_Eout_Win)-len(common_elements_Eout_Win_check)
        common_elements_Eout_Win_prev= common_elements_Eout_Win.copy()
    if common_elements_Eout_Sin:
        common_elements_Eout_Sin_check= common_elements_Eout_Sin.intersection(common_elements_Eout_Sin_prev)
        e2s+=len(common_elements_Eout_Sin)-len(common_elements_Eout_Sin_check)
        common_elements_Eout_Sin_prev= common_elements_Eout_Sin.copy()
    if common_elements_Eout_Ein:
        common_elements_Eout_Ein_check =common_elements_Eout_E.intersection(common_elements_Eout_Ein_prev)
        e2e+=len(common_elements_Eout_Ein)-len(common_elements_Eout_Ein_check)
        common_elements_Eout_Ein_prev= common_elements_Eout_Ein.copy()
        
    
     ##Check if the element is common in West out and S,E,N in boxes##
        
    common_elements_Wout_Nin=areaWout.intersection(areaNin)
    common_elements_Wout_Sin=areaWout.intersection(areaSin)
    common_elements_Wout_Ein=areaWout.intersection(areaEin)
    common_elements_Wout_Win=areaWout.intersection(areaWin)
    
    
    if common_elements_Wout_Nin:
        common_elements_Wout_Nin_check= common_elements_Wout_Nin.intersection(common_elements_Wout_Nin_prev)
        w2n+=len(common_elements_Wout_Nin)- len(common_elements_Wout_Nin_check)
        common_elements_Wout_Nin_prev = common_elements_Wout_Nin.copy()
    if common_elements_Wout_Ein:
        common_elements_Wout_Ein_check= common_elements_Wout_Ein.intersection(common_elements_Wout_Ein_prev)
        w2e+=len(common_elements_Wout_Ein)- len(common_elements_Wout_Ein_check)
        common_elements_Wout_Ein_prev= common_elements_Wout_Ein.copy()
    if common_elements_Wout_Sin:
        common_elements_Wout_Sin_check= common_elements_Wout_Sin.intersection(common_elements_Wout_Sin_prev)
        w2s+=len(common_elements_Wout_Sin)-len(common_elements_Wout_Sin_check)
        common_elements_Wout_Sin_prev = common_elements_Wout_Sin.copy()
    if common_elements_Wout_Win:
        common_elements_Wout_Win_check = common_elements_Wout_Win.intersection(common_elements_Wout_Win_prev)
        w2w+=len(common_elements_Wout_Win)-len(common_elements_Wout_Win_check)
        common_elements_Wout_Win_prev = common_elements_Wout_Win.copy()
            
    
    ######For North to other direction text monitor######   
    p=len(areaEin)   
    cv2.putText(frame,str("N2E"+":"+str(n2e)),(20,30),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("N2S"+":"+str(n2s)),(132,30),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("N2W"+":"+str(n2w)),(244,30),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("N2N"+":"+str(n2n)),(365,30),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    
    ######For South to other direction text  on the monitor######
    cv2.putText(frame,str("S2E"+":"+str(s2e)),(20,56),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("S2S"+":"+str(s2s)),(132,56),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("S2W"+":"+str(s2w)),(244,56),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("S2N"+":"+str(s2n)),(365,56),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    
    ######For East to other direction text on the monitor######
    
    cv2.putText(frame,str("E2E"+":"+str(e2e)),(20,82),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("E2S"+":"+str(e2s)),(132,82),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("E2W"+":"+str(e2w)),(244,82),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("E2N"+":"+str(e2n)),(365,82),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    
    ######For West to other direction text on the monitor######
    
    cv2.putText(frame,str("W2E"+":"+str(w2e)),(20,108),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("W2S"+":"+str(w2s)),(132,108),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("W2W"+":"+str(w2w)),(244,108),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    cv2.putText(frame,str("W2N"+":"+str(w2n)),(365,108),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,269),2)
    
    
    
    ##print("Tracking objects")
    ##print(tracking_objects)
    
                
    #print("CUR FRAME LEFT PTS")
    #print(center_points_cur_frame)
    
   # print ("PREV FRAME")
    #print(center_points_prev_frame)
        
    # Display frame
    cv2.imshow('FRAME', frame)
    
    #Make a copy of the current frame
    center_points_prev_frame = center_points_cur_frame.copy()
    
    if cv2.waitKey(0) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()

    
    
