# importing libraries
import cv2,time
from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv


mot_tracker = Sort()
vehicles = [2, 3, 5, 7]
# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('sample.mp4')
print(cap.get(cv2.CAP_PROP_FPS))
cap.set(cv2.CAP_PROP_FPS,10) 

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")
cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# Read until video is completed
rate_per_sec=5
prev=0
while(cap.isOpened()):
    
# Capture frame-by-frame
    ret, _frame = cap.read()
    
    
    if ret:
        
        if time.time()>prev+1/rate_per_sec:
            prev=time.time()
            """
            detections = coco_model(_frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    cv2.rectangle(_frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3)
                    detections_.append([x1, y1, x2, y2, score])
            """
            license_plates = license_plate_detector(_frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                if score>0.5:
                    cv2.rectangle(_frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),6)
                # crop license plate
                    license_plate_crop = _frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    if license_plate_text is not None and  float(license_plate_text_score)>0.5:
                        (text_width, text_height), _ = cv2.getTextSize(
                        license_plate_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        10)
                        cv2.putText(_frame,
                                "{} [{}]".format(license_plate_text,round(license_plate_text_score*100)/100),
                                (int((x1 + x2 - text_width) / 2), int(y1  - 50 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4,
                                (255, 0, 0),
                              10)
                     
    # Display the resulting frame
        frame = cv2.resize(_frame, (960, 540))
        cv2.imshow('output', frame)
        
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()