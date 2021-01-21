##########################################################################################################
#                                                                                                        #
#   Object Detection (video) From TF2 Saved Model                                                        #
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
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(conf["model"])
end_time = time.time()
elapsed_time = end_time - start_time
print("[INFO] model loaded ... took {} seconds".format(elapsed_time))

# Load labelmap
category_index = label_map_util.create_category_index_from_labelmap(conf["label"],
                                                                    use_display_name=True)

'''
Input video 
'''
print("[INFO] loading video from file ...")

# load input video
vs = cv2.VideoCapture(conf["video_input"])

# prepare variable for writer that we will use to write processed frames
writer = None
# prepare variables for spatial dimensions of the frames
h, w = None, None

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
    # read the next frame from the file
    grabbed, frame = vs.read()
    
    # if the frame was not grabbed, then end of the stream
    if not grabbed:
        break

    # get spatial dimensions of the frame (only 1st time)
    if w is None or h is None:
        h, w = frame.shape[:2]
        
    frame_np = np.array(frame)

    frame_height, frame_width = frame.shape[:2]

    start = time.time()
    
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

            #Extract the detected number plate
            if object_name == "plate":
                licence_img = frame_np[ymin:ymax, xmin:xmax]
                text = ocr_plate_recognition.recognize_plate(licence_img)
                cv2.putText(frame_np, text, (xmin, ymax + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 0), 2)
                #cv2.imwrite('matricula_reconocida.jpg', licence_img)

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
        writer = cv2.VideoWriter(conf["video_output"], fourcc, 32, (frame_np.shape[1], frame_np.shape[0]), True)
        
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write processed current frame to the file
    writer.write(frame_np)

    key = cv2.waitKey(1) & 0xFF
    # if the "Esc" key was pressed, break from the loop
    if key == (ord("q")) or key == 27:
        break

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