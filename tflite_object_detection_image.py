##########################################################################################################
#                                                                                                        #
#   Object Detection (image) From TF2 with TensorFLow Lite                                               #
#   file: tflite_object_detection_image.py                                                               #
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

end_time = time.time()
elapsed_time = end_time - start_time
print("[INFO] model loaded ... took {} seconds".format(elapsed_time))

# LOAD LABELS
with open(conf["tflite_label"], 'r') as f:
    category_index = [line.strip() for line in f.readlines()]

'''
Image
'''
input_mean = 127.5
input_std = 127.5

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# load input image
image = cv2.imread(conf["image_input"])
image_height, image_width = image.shape[:2]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)
print("[INFO] image loaded from file ...")

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
        ymin = int(max(1,(nms_boxes[i][0] * image_height)))
        xmin = int(max(1,(nms_boxes[i][1] * image_width)))
        ymax = int(min(image_height,(boxes[i][2] * image_height)))
        xmax = int(min(image_width,(boxes[i][3] * image_width)))
                
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        
        # Draw label
        # Look up object name from "labels" array using class index
        object_name = category_index[int(classes[i])]
        label = '%s: %d%%' % (object_name, int(scores[i]*100))
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (10, 255, 0), cv2.FILLED)
        
        # Draw white box to put label text in
        cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Draw label text   
        
        #Extract the detected number plate
        if object_name == "licence":
            licence_img = image[ymin:ymax, xmin:xmax]
            text = ocr_plate_recognition.recognize_plate(licence_img)
            cv2.putText(image, text, (xmin, ymax + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)

'''
Display image
'''
# show image
cv2.namedWindow("Image Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Image Detections", image)

# write new image with detections
cv2.imwrite(conf["image_output"], image)

# wait for any key being pressed
cv2.waitKey(0)
# destroy opened window
cv2.destroyWindow("Image Detections") 