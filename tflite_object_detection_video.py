##########################################################################################################
#                                                                                                        #
#   Object Detection (image) From TF2 with TensorFLow Lite                                               #
#   file: tflite_object_detection_video.py                                                               #
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
interpreter = tflite.Interpreter(model_path=conf["tflite_model"], 
                                 experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
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
print("[INFO] loading video from file ...")

# load input video
vs = cv2.VideoCapture(conf["video_input"])
frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# prepare variable for writer that we will use to write processed frames
writer = None

# try to determine the total number of frames in the video file
try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

'''
Read frames in the loop
'''
# variable for counting frames
f = 0
# variable for counting total time
t = 0

# loop over frames from the video file stream
while True:
    '''
    Frame
    '''
    # read the next frame from the file
    grabbed, frame = vs.read()
    
    # if the frame was not grabbed, then end of the stream
    if not grabbed:
        break
        
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
            ymin = int(max(1,(nms_boxes[i][0] * frame_height)))
            xmin = int(max(1,(nms_boxes[i][1] * frame_width)))
            ymax = int(min(frame_height,(boxes[i][2] * frame_height)))
            xmax = int(min(frame_width,(boxes[i][3] * frame_width)))

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
    
    end = time.time()
    
    # increase counters for frames and total time
    f += 1
    t += end - start
    
    '''
    Write processed frame into file
    '''
    if writer is None:
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*conf["video_codec"])
        writer = cv2.VideoWriter(conf["video_output"], fourcc, 32, (frame_width, frame_height), True)
        
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write processed current frame to the file
    writer.write(frame)


'''
Finish
'''
# print final results
print()
print("[INFO] total number of frames", f)
print("[INFO] total amount of time {:.5f} seconds".format(t))
print("[INFO] fps:", round((f / t), 1))
print()

# do a bit of cleanup
print("[INFO] cleaning up...")
# release video reader and writer
cv2.destroyAllWindows()
vs.release()
writer.release()