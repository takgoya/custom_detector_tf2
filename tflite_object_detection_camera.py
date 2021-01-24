##########################################################################################################
#                                                                                                        #
#   Object Detection (image) From TF2 with TensorFLow Lite                                               #
#   file: tflite_object_detection_camera.py                                                               #
#                                                                                                        #
#   Author: Javier Goya Pérez                                                                            #
#   Date: January 2021                                                                                   #
#                                                                                                        #
##########################################################################################################

# This code is based on the TensorFlow2 Object Detection API tutorial
# (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html#putting-everything-together)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)

import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate

import tensorflow as tf

import argparse
import json
import cv2
import numpy as np

import sys
import time

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# utils library from https://www.pyimagesearch.com (author: Adrian Rosebrock)
from imutils.video import VideoStream
from imutils.video import FPS

# utils functions for tesseract ocr plate recognition
import ocr_plate_recognition

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
Load model and labels
'''
print("[INFO] loading model ...")
start_time = time.time()

# LOAD TFLITE MODEL
use_tpu = conf["use_tpu"]
if use_tpu == 1:
    interpreter = tflite.Interpreter(model_path=conf["tflite_model_tpu"], 
                                     experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = tflite.Interpreter(model_path=conf["tflite_model"])

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
    
input_mean = 127.5
input_std = 127.5

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

end_time = time.time()
elapsed_time = end_time - start_time
print("[INFO] model loaded ... took {} seconds".format(elapsed_time))

# LOAD LABELS
with open(conf["tflite_label"], 'r') as f:
    category_index = [line.strip() for line in f.readlines()]

'''
Input video 
'''
# initialize the video stream and allow the camera
# sensor to warmup
vs = VideoStream(usePiCamera=conf["use_picamera"],
                 resolution=tuple(conf["resolution"]),
                 framerate=conf["fps"]).start()

print("[INFO] warming up camera...")
time.sleep(conf["camera_warmup_time"])
fps = FPS().start()
    
# prepare variable for writer that we will use to write processed frames
writer = None
# prepare variables for spatial dimensions of the frames
h, w = None, None
print("[INFO] starting video from camera ...")

# prepare variable for writer that we will use to write processed frames
writer = None

'''
Read frames in the loop
'''
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    frame = vs.read()
    
    # get spatial dimensions of the frame (only 1st time)
    if w is None or h is None:
        h, w = frame.shape[:2]
        
    start = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    '''
    Run inference
    '''
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    floating_model = (input_details[0]['dtype'] == np.float32)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    detections = {'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                  'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                  'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                  'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]}

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    scores = detections['detection_scores'] # Confidence of detected objects
    boxes = detections['detection_boxes']  # Bounding box coordinates of detected objects
    classes = detections['detection_classes'] # Class index of detected objects

    # Apply Non Max Suppression
    length = len([i for i in detections['detection_scores'] if i>conf["confidence"]])
    nms_indices = tf.image.non_max_suppression(boxes, scores, length, conf["threshold"])
    nms_boxes = tf.gather(boxes, nms_indices)

    for i in range(len(scores)):
        if ((scores[i] > conf["confidence"]) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions
            # need to force them to be within image using max() and min()
            ymin = int(max(1,(nms_boxes[i][0] * height)))
            xmin = int(max(1,(nms_boxes[i][1] * width)))
            ymax = int(min(height,(boxes[i][2] * height)))
            xmax = int(min(width,(boxes[i][3] * width)))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            # Look up object name from "labels" array using class index
            object_name = category_index[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (10, 255, 0), cv2.FILLED)

            # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Draw label text   

            #Extract the detected number plate
            if object_name == "licence":
                licence_img = frame[ymin:ymax, xmin:xmax]
                text = ocr_plate_recognition.recognize_plate(licence_img)
                cv2.putText(frame, text, (xmin, ymax + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
    
    cv2.namedWindow("Camera Detections", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera Detections", frame)
    
    fps.update()

    if writer is None:
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*conf["video_codec"])
        writer = cv2.VideoWriter(conf["video_camera_output"], fourcc, 16, (frame.shape[1], frame.shape[0]), True)

    # write processed current frame to the file
    writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    # if the "Esc" key was pressed, break from the loop
    if key == (ord("q")) or key == 27:
        break

'''
Finish
'''
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
print("[INFO] cleaning up...")
# release video reader and writer
cv2.destroyAllWindows()
vs.stop()
writer.release()