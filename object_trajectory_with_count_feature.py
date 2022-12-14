from absl import flags
import sys
Fl = flags.FLAGS #Needed for yolov3 flag settings
Fl(sys.argv)

import numpy as np
import time #to tracking the frames per second
import cv2 # for visualization of object trakcing
import matplotlib.pyplot as plt #Color mapping
from tensorflow import summary
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes # convert the boxes back to deepsort format
from deep_sort import preprocessing     #used for NMS
from deep_sort import nn_matching #For the association matrix
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet



#initialize the 80 classes

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
#Load the model and weights
model =YoloV3(classes=len(class_names))
# model.summary()
model.load_weights('./weights/yolov3.tf')

#Intialize the DEEPSORT

max_cosine_dist = 0.5 # A threshold to analyze whether the previous frame is similar to the current frame
nn_bget = None # To store the feature vectors
nms_max_ovlp = 0.8 # Default is usually 1 but there can be similar kind of detection for the same object which we do not want!

model_filename ='model_data/mars-small128.pb' #pretrained CNN for tracking peds 

encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine',max_cosine_dist,nn_bget)

tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/test.mp4')

#video result Saving
codec = cv2.VideoWriter_fourcc(*'XVID') #for .avi format
vid_fps = int(vid.get(cv2.CAP_PROP_FPS)) #gives us the original FPS
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('./data/video/Results_person_count.avi', codec, vid_fps, (vid_width, vid_height))

from collections import deque
pts = [deque(maxlen=30) for _ in range(1000)] #For the historical trajectory datapoints to be saved also when new trajectory points are saved after maxlen the oldest ones are removed

C =[] #it is a C for number of persons
while True:
    _,img = vid.read()
    if img is None:
        print('Completed')
        break
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # The cv2 color code is BGR so we need to convert to RGB as that is what yolo is expecting
    img_in = tf.expand_dims(img_in,0) # image originally has 3 dimensions but we need one for batch size so adding one more to the original
    img_in = transform_images(img_in,416)

    t1 = time.time()

    boxes, scores, classes, nums = model.predict(img_in,steps=1)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]                         #  So the nd arrays are in tlwh format which is topleft X and Y with width and height

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_ovlp, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections) # updates the kahlman tracker parameter and compare the previous and current target

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
    current_count = int(0)
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1: #skip if there is no update
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)] #different colors for different bounding
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2) #Providing the top left and bottom right corners co-ordinates
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)      
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)                    #x-axis should have enough positions where it can contain the tracks and track ID

        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2)) #to find the center co-oridnates of the tracked object
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:          #if the previous track has no track and current has none continue
                continue
            thickness = int(np.sqrt(81/float(j+1))*2) #closer lines are thinner and farther point line is thicker
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = img.shape
        cv2.line(img, (0, int(3*height/6+height/25)), (width, int(3*height/6+height/25)), (0, 255, 255), thickness=2)
        cv2.line(img, (0, int(3*height/6-height/25)), (width, int(3*height/6-height/25)), (0, 255, 255), thickness=2)

        center_track = int(((bbox[1])+(bbox[3]))/2) #center of each tracked object

        if center_track <= int(3*height/6+height/25) and center_track >= int(3*height/6-height/25):
            if class_name == 'person':
                C.append(int(track.track_id))
                current_count += 1

    total_count = len(set(C))
    cv2.putText(img, "Current Person Count in One Area: " + str(current_count), (0, 80), 0, 1, (0, 255, 0), 2)
    cv2.putText(img, "Total Person Count: " + str(total_count), (0,130), 0, 1, (0,255,0), 2) 

    fps = 1./(time.time()-t1)
    cv2.putText(img,"FPS: {:.2f}".format(fps), (0,30), 0, 1, (0, 0, 255), 2)
    cv2.resizeWindow('output',1024,768)
    cv2.imshow('output',img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()







