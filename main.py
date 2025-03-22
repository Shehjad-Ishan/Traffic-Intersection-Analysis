import cv2
import torch
import numpy as np
import math

# Kalman Filter Implementation
class KalmanFilter:
    def __init__(self, dt=1):
        # State: [x, y, vx, vy]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        
        # Transition matrix
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * 0.03
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], np.float32) * 1
        
    def predict(self):
        return self.kalman.predict()
    
    def update(self, measurement):
        return self.kalman.correct(measurement)
    
    def init_position(self, measurement):
        # Initialize state with measurement
        self.kalman.statePre = np.array([[measurement[0]], [measurement[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[measurement[0]], [measurement[1]], [0], [0]], np.float32)

# Tracking object with Kalman filter
class TrackedObject:
    def __init__(self, object_id, position):
        self.id = object_id
        self.position = position
        self.kalman = KalmanFilter()
        self.kalman.init_position(np.array([position[0], position[1]], np.float32))
        self.missed_frames = 0
        self.max_missed_frames = 5  # Maximum number of frames to keep predicting without measurements
        
    def predict(self):
        predicted = self.kalman.predict()
        return (int(predicted[0][0]), int(predicted[1][0]))
    
    def update(self, new_position):
        measurement = np.array([new_position[0], new_position[1]], np.float32)
        corrected = self.kalman.update(measurement)
        self.position = (int(corrected[0][0]), int(corrected[1][0]))
        self.missed_frames = 0
        return self.position
    
    def has_disappeared(self):
        return self.missed_frames > self.max_missed_frames

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='projectweight.pt', force_reload=True)
if not model:
    print("Error loading model, falling back to YOLOv5s")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open video
cap = cv2.VideoCapture('./traf720.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

# Declare variables
tracked_objects = {}
track_id = 0
count = 0

# Define regions for traffic flow analysis
area_nrt_in = [(308, 282), (331, 288), (340, 243), (321, 240)]
area_nrt_out = [(340, 243), (321, 240), (330, 206), (350, 207)]
area_est_in = [(413, 136), (403, 168), (528, 151), (519, 121)]
area_est_out = [(700, 172), (726, 154), (799, 185), (778, 211)]
area_sth_out = [(798, 235), (748, 299), (785, 311), (827, 252)]
area_sth_in = [(712, 358), (748, 299), (785, 311), (744, 366)]
area_wst_out = [(319, 312), (467, 354), (448, 389), (310, 351)]
area_wst_in = [(468, 354), (617, 399), (603, 433), (448, 390)]

# Sets to track objects in each area
areaNin = set()
areaNout = set()
areaSout = set()
areaSin = set()
areaEin = set()
areaEout = set()
areaWin = set()
areaWout = set()

# Counters for traffic flow directions
n2n, n2s, n2e, n2w = 0, 0, 0, 0
s2s, s2n, s2e, s2w = 0, 0, 0, 0
e2e, e2s, e2w, e2n = 0, 0, 0, 0
w2w, w2e, w2n, w2s = 0, 0, 0, 0

# Sets for tracking common elements between areas
common_elements = {
    'Nout_Ein': set(), 'Nout_Win': set(), 'Nout_Sin': set(), 'Nout_Nin': set(),
    'Sout_Nin': set(), 'Sout_Win': set(), 'Sout_Ein': set(), 'Sout_Sin': set(),
    'Eout_Nin': set(), 'Eout_Sin': set(), 'Eout_Win': set(), 'Eout_Ein': set(),
    'Wout_Nin': set(), 'Wout_Sin': set(), 'Wout_Ein': set(), 'Wout_Win': set()
}

common_elements_prev = {
    'Nout_Ein': set(), 'Nout_Win': set(), 'Nout_Sin': set(), 'Nout_Nin': set(),
    'Sout_Nin': set(), 'Sout_Win': set(), 'Sout_Ein': set(), 'Sout_Sin': set(),
    'Eout_Nin': set(), 'Eout_Sin': set(), 'Eout_Win': set(), 'Eout_Ein': set(),
    'Wout_Nin': set(), 'Wout_Sin': set(), 'Wout_Ein': set(), 'Wout_Win': set()
}

# Object detection and tracking
while True:
    detections_current_frame = []
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    frame = cv2.resize(frame, (1020, 500))
    
    # Draw region polygons
    cv2.polylines(frame, [np.array(area_nrt_in, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area_nrt_out, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area_wst_out, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area_wst_in, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area_sth_in, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area_sth_out, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area_est_in, np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(area_est_out, np.int32)], True, (0, 255, 0), 1)
    
    # Run object detection
    results = model(frame)
    count += 1
    
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        det_class = str(row['name'])
        
        # Calculate centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detections_current_frame.append((cx, cy))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Predict new positions for all tracked objects
    for object_id, tracked_obj in list(tracked_objects.items()):
        predicted_position = tracked_obj.predict()
        
        # Find if there is a matching detection
        object_exists = False
        best_match_idx = -1
        min_distance = float('inf')
        
        for i, detection in enumerate(detections_current_frame):
            distance = math.hypot(detection[0] - predicted_position[0], 
                                  detection[1] - predicted_position[1])
            if distance < 30 and distance < min_distance:  # Increased threshold from 20 to 30
                min_distance = distance
                best_match_idx = i
                object_exists = True
        
        # Update tracked object if match found
        if object_exists:
            detection = detections_current_frame[best_match_idx]
            tracked_objects[object_id].update(detection)
            detections_current_frame.pop(best_match_idx)  # Remove assigned detection
        else:
            # Increment missed frames counter
            tracked_objects[object_id].missed_frames += 1
            # Remove object if it has been missing for too long
            if tracked_objects[object_id].has_disappeared():
                tracked_objects.pop(object_id)
    
    # Add new tracked objects for unassigned detections
    for detection in detections_current_frame:
        tracked_objects[track_id] = TrackedObject(track_id, detection)
        track_id += 1
    
    # Draw tracked objects and update area sets
    for object_id, tracked_obj in tracked_objects.items():
        position = tracked_obj.position
        
        # Draw circle and ID
        cv2.circle(frame, position, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (position[0], position[1] - 7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Test which areas the object is in
        ptx, pty = int(position[0]), int(position[1])
        
        # Check if point is inside each polygon
        resNin = cv2.pointPolygonTest(np.array(area_nrt_in, np.int32), (ptx, pty), False)
        resEin = cv2.pointPolygonTest(np.array(area_est_in, np.int32), (ptx, pty), False)
        resWin = cv2.pointPolygonTest(np.array(area_wst_in, np.int32), (ptx, pty), False)
        resSin = cv2.pointPolygonTest(np.array(area_sth_in, np.int32), (ptx, pty), False)
        resNout = cv2.pointPolygonTest(np.array(area_nrt_out, np.int32), (ptx, pty), False)
        resSout = cv2.pointPolygonTest(np.array(area_sth_out, np.int32), (ptx, pty), False)
        resEout = cv2.pointPolygonTest(np.array(area_est_out, np.int32), (ptx, pty), False)
        resWout = cv2.pointPolygonTest(np.array(area_wst_out, np.int32), (ptx, pty), False)
        
        # Update area sets
        if resNin > 0:
            areaNin.add(object_id)
        if resNout > 0:
            areaNout.add(object_id)
        if resSin > 0:
            areaSin.add(object_id)
        if resSout > 0:
            areaSout.add(object_id)
        if resEin > 0:
            areaEin.add(object_id)
        if resEout > 0:
            areaEout.add(object_id)
        if resWin > 0:
            areaWin.add(object_id)
        if resWout > 0:
            areaWout.add(object_id)
    
    # Check common elements between areas
    common_elements['Nout_Ein'] = areaNout.intersection(areaEin)
    common_elements['Nout_Win'] = areaNout.intersection(areaWin)
    common_elements['Nout_Sin'] = areaNout.intersection(areaSin)
    common_elements['Nout_Nin'] = areaNout.intersection(areaNin)
    
    # Update North-to-X counts
    if common_elements['Nout_Ein']:
        check = common_elements['Nout_Ein'].intersection(common_elements_prev['Nout_Ein'])
        n2e += len(common_elements['Nout_Ein']) - len(check)
        common_elements_prev['Nout_Ein'] = common_elements['Nout_Ein'].copy()
    
    if common_elements['Nout_Win']:
        check = common_elements['Nout_Win'].intersection(common_elements_prev['Nout_Win'])
        n2w += len(common_elements['Nout_Win']) - len(check)
        common_elements_prev['Nout_Win'] = common_elements['Nout_Win'].copy()
    
    if common_elements['Nout_Sin']:
        check = common_elements['Nout_Sin'].intersection(common_elements_prev['Nout_Sin'])
        n2s += len(common_elements['Nout_Sin']) - len(check)
        common_elements_prev['Nout_Sin'] = common_elements['Nout_Sin'].copy()
    
    if common_elements['Nout_Nin']:
        check = common_elements['Nout_Nin'].intersection(common_elements_prev['Nout_Nin'])
        n2n += len(common_elements['Nout_Nin']) - len(check)
        common_elements_prev['Nout_Nin'] = common_elements['Nout_Nin'].copy()
    
    # South-to-X common elements
    common_elements['Sout_Nin'] = areaSout.intersection(areaNin)
    common_elements['Sout_Win'] = areaSout.intersection(areaWin)
    common_elements['Sout_Ein'] = areaSout.intersection(areaEin)
    common_elements['Sout_Sin'] = areaSout.intersection(areaSin)
    
    # Update South-to-X counts
    if common_elements['Sout_Nin']:
        check = common_elements['Sout_Nin'].intersection(common_elements_prev['Sout_Nin'])
        s2n += len(common_elements['Sout_Nin']) - len(check)
        common_elements_prev['Sout_Nin'] = common_elements['Sout_Nin'].copy()
    
    if common_elements['Sout_Win']:
        check = common_elements['Sout_Win'].intersection(common_elements_prev['Sout_Win'])
        s2w += len(common_elements['Sout_Win']) - len(check)
        common_elements_prev['Sout_Win'] = common_elements['Sout_Win'].copy()
    
    if common_elements['Sout_Ein']:
        check = common_elements['Sout_Ein'].intersection(common_elements_prev['Sout_Ein'])
        s2e += len(common_elements['Sout_Ein']) - len(check)
        common_elements_prev['Sout_Ein'] = common_elements['Sout_Ein'].copy()
    
    if common_elements['Sout_Sin']:
        check = common_elements['Sout_Sin'].intersection(common_elements_prev['Sout_Sin'])
        s2s += len(common_elements['Sout_Sin']) - len(check)
        common_elements_prev['Sout_Sin'] = common_elements['Sout_Sin'].copy()
    
    # East-to-X common elements
    common_elements['Eout_Nin'] = areaEout.intersection(areaNin)
    common_elements['Eout_Sin'] = areaEout.intersection(areaSin)
    common_elements['Eout_Win'] = areaEout.intersection(areaWin)
    common_elements['Eout_Ein'] = areaEout.intersection(areaEin)
    
    # Update East-to-X counts
    if common_elements['Eout_Nin']:
        check = common_elements['Eout_Nin'].intersection(common_elements_prev['Eout_Nin'])
        e2n += len(common_elements['Eout_Nin']) - len(check)
        common_elements_prev['Eout_Nin'] = common_elements['Eout_Nin'].copy()
    
    if common_elements['Eout_Win']:
        check = common_elements['Eout_Win'].intersection(common_elements_prev['Eout_Win'])
        e2w += len(common_elements['Eout_Win']) - len(check)
        common_elements_prev['Eout_Win'] = common_elements['Eout_Win'].copy()
    
    if common_elements['Eout_Sin']:
        check = common_elements['Eout_Sin'].intersection(common_elements_prev['Eout_Sin'])
        e2s += len(common_elements['Eout_Sin']) - len(check)
        common_elements_prev['Eout_Sin'] = common_elements['Eout_Sin'].copy()
    
    if common_elements['Eout_Ein']:
        check = common_elements['Eout_Ein'].intersection(common_elements_prev['Eout_Ein'])
        e2e += len(common_elements['Eout_Ein']) - len(check)
        common_elements_prev['Eout_Ein'] = common_elements['Eout_Ein'].copy()
    
    # West-to-X common elements
    common_elements['Wout_Nin'] = areaWout.intersection(areaNin)
    common_elements['Wout_Sin'] = areaWout.intersection(areaSin)
    common_elements['Wout_Ein'] = areaWout.intersection(areaEin)
    common_elements['Wout_Win'] = areaWout.intersection(areaWin)
    
    # Update West-to-X counts
    if common_elements['Wout_Nin']:
        check = common_elements['Wout_Nin'].intersection(common_elements_prev['Wout_Nin'])
        w2n += len(common_elements['Wout_Nin']) - len(check)
        common_elements_prev['Wout_Nin'] = common_elements['Wout_Nin'].copy()
    
    if common_elements['Wout_Ein']:
        check = common_elements['Wout_Ein'].intersection(common_elements_prev['Wout_Ein'])
        w2e += len(common_elements['Wout_Ein']) - len(check)
        common_elements_prev['Wout_Ein'] = common_elements['Wout_Ein'].copy()
    
    if common_elements['Wout_Sin']:
        check = common_elements['Wout_Sin'].intersection(common_elements_prev['Wout_Sin'])
        w2s += len(common_elements['Wout_Sin']) - len(check)
        common_elements_prev['Wout_Sin'] = common_elements['Wout_Sin'].copy()
    
    if common_elements['Wout_Win']:
        check = common_elements['Wout_Win'].intersection(common_elements_prev['Wout_Win'])
        w2w += len(common_elements['Wout_Win']) - len(check)
        common_elements_prev['Wout_Win'] = common_elements['Wout_Win'].copy()
    
    # Display traffic counts
    cv2.putText(frame, f"N2E: {n2e}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"N2S: {n2s}", (132, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"N2W: {n2w}", (244, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"N2N: {n2n}", (365, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    
    cv2.putText(frame, f"S2E: {s2e}", (20, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"S2S: {s2s}", (132, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"S2W: {s2w}", (244, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"S2N: {s2n}", (365, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    
    cv2.putText(frame, f"E2E: {e2e}", (20, 82), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"E2S: {e2s}", (132, 82), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"E2W: {e2w}", (244, 82), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"E2N: {e2n}", (365, 82), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    
    cv2.putText(frame, f"W2E: {w2e}", (20, 108), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"W2S: {w2s}", (132, 108), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"W2W: {w2w}", (244, 108), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    cv2.putText(frame, f"W2N: {w2n}", (365, 108), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 269), 2)
    
    # Display frame
    cv2.imshow('FRAME', frame)
    
    # Use waitKey(1) for continuous playback instead of waitKey(0)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()
