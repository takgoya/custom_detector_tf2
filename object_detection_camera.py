##########################################################################################################
#                                                                                                        #
#   Object Detection (camera) From TF2 Saved Model                                                       #
#   file: object_detection_image.py                                                                      #
#                                                                                                        #
#   Author: Javier Goya Pérez                                                                            #
#   Date: January 2021                                                                                   #
#                                                                                                        #
##########################################################################################################

# This code is based on the TensorFlow2 Object Detection API tutorial
# (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html#putting-everything-together)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logging (2)

import cv2
import numpy as np

import matplotlib
import warnings
warnings.filterwarnings('ignore') # Suppress Matplotlib warningsfrom collections import defaultdict

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# utils library from https://www.pyimagesearch.com (author: Adrian Rosebrock)
from imutils.video import VideoStream
from imutils.video import FPS

import argparse
import json
import pathlib
import time
import datetime

# utils functions for tesseract ocr plate recognition
import ocr_plate_recognition

# utils functions for db
from db import db_utils

# utils functions for gps
from gps import gps_utils

'''
Arguments
'''
# json
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())
# load the configuration
conf = json.load(open(args["conf"]))

'''
GPS
'''
if (conf["use_gps"]):
    gps_socket, data_stream = gps_utils.init_gps()
    
'''
Load model and labels
'''
print("[TF] loading model ...")
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(conf["model"])
end_time = time.time()
elapsed_time = end_time - start_time
print("[TF] model loaded ... took {} seconds".format(elapsed_time))

# Load labelmap
category_index = label_map_util.create_category_index_from_labelmap(conf["label"],
                                                                    use_display_name=True)

'''
Input video 
'''
# initialize the video stream and allow the camera
# sensor to warmup
vs = VideoStream(usePiCamera=conf["use_picamera"],
                 resolution=tuple(conf["resolution"]),
                 framerate=conf["fps"]).start()

print("[TF] warming up camera...")
time.sleep(conf["camera_warmup_time"])
fps = FPS().start()
    
# prepare variable for writer that we will use to write processed frames
writer = None
# prepare variables for spatial dimensions of the frames
h, w = None, None
print("[TF] starting video from camera ...")

'''
Prepare DB
'''
# Create (if not exists) DB connection
conn = db_utils.create_connection(conf["db"])

if conn != None:
    # Create (if not exists) RECORDINGS table
    db_utils.create_recordings_table(conn)
    # Create (if not exists) DETECTIONS table
    db_utils.create_detections_table(conn)
    print("[TF] DB configured")
else:
    print("[TF] error while configuring DB")

# Generate recording entry name
recording_name = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
# Insert recording into RECORDINGS table
recording_id = db_utils.insert_recording(conn, recording_name)
    
'''
Input video 
'''
# Name for generated videofile
recording_path = conf["video_camera_output"] + "/" + recording_name + ".avi" 

# variable for counting frames
f_count = 0

# variable for counting time
start_time = time.time()

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    frame = vs.read()
    
    # get spatial dimensions of the frame (only 1st time)
    if w is None or h is None:
        h, w = frame.shape[:2]
        
    frame_np = np.array(frame)
    
    '''
    Run inference
    '''
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame_np)
    # The model expects a batch of frames, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(frame_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    scores = detections['detection_scores'] # Bounding box coordinates of detected objects
    boxes = detections['detection_boxes'] # Confidence of detected objects
    classes = detections['detection_classes'] # Class index of detected objects

    # Apply Non Max Suppression
    length = len([i for i in detections['detection_scores'] if i>conf["confidence"]])
    nms_indices = tf.image.non_max_suppression(boxes, scores, length, conf["threshold"])
    nms_boxes = tf.gather(boxes, nms_indices)
    
    # dictionary for db
    db_detections = {}
    
    for i in range(len(scores)):
        if ((scores[i] > conf["threshold"]) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of frame dimensions
            # need to force them to be within frame using max() and min()
            ymin = int(max(1,(nms_boxes[i][0] * frame_height)))
            xmin = int(max(1,(nms_boxes[i][1] * frame_width)))
            ymax = int(min(frame_height,(nms_boxes[i][2] * frame_height)))
            xmax = int(min(frame_width,(nms_boxes[i][3] * frame_width)))

            cv2.rectangle(frame_np, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            # Look up object name from "labels" array using class index
            object_name = category_index[int(classes[i])]['name']
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            #label = "%s" % (object_name)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame_np, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (10, 255, 0), cv2.FILLED)

            # Draw white box to put label text in
            cv2.putText(frame_np, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Draw label text           '''

            plate_num = ""
            #Extract the detected number plate
            if object_name == "licence":
                licence_img = frame_np[ymin:ymax, xmin:xmax]
                image_h, image_w = licence_img.shape[:2]
                if image_w != 0 and image_h != 0:
                    plate_num = ocr_plate_recognition.recognize_plate(licence_img)
                    cv2.putText(frame_np, plate_num, (xmin, ymax + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
                    if plate_num != "":
                        print("[TF] licence recognition = {}".format(plate_num))
                        if (i-1) >= 0:
                            db_detections[plate_num] = category_index[int(classes[i-1])]['name']
                        else:
                            db_detections[plate_num] = category_index[int(classes[i+1])]['name']

    cv2.namedWindow("Camera Detections", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera Detections", frame_np)
    
    fps.update()
    
    elapsed_time = round(time.time() - start_time, 2)
    
    # increase counters for frames and total time
    f_count += 1    

    '''
    Write processed frame into file
    '''
    if writer is None:
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*conf["video_codec"])
        writer = cv2.VideoWriter(recording_path, fourcc, 16, (frame.shape[1], frame.shape[0]), True)

    # write processed current frame to the file
    writer.write(frame)

    '''
    Insert into DB
    '''
    # get gps position
    gps_lat = gps_lon = 0
    if (conf["use_gps"]):
        gps_lat, gps_lon = gps_utils.get_position(gps_socket, data_stream)

    # get detection time
    detection_datetime = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

    # add entry to db
    for key, value in db_detections.items():
        detection = (recording_id, value, key, gps_lat, gps_lon, elapsed_time, f_count, detection_datetime)
        db_utils.insert_detection(conn, detection)

    '''
    Break from loop
    '''
    key = cv2.waitKey(1) & 0xFF
    # if the "Esc" key was pressed, break from the loop
    if key == (ord("q")) or key == 27:
        break

'''
Finish
'''
end_time = time.time()

# stop the timer and display FPS information
fps.stop()
print("[TF] elasped time: {:.2f}".format(fps.elapsed()))
print("[TF] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
print("[TF] cleaning up...")
# release video reader and writer
cv2.destroyAllWindows()
vs.stop()
writer.release()