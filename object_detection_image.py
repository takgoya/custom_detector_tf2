import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf
import pathlib

from collections import defaultdict

#from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from imutils.video import VideoStream
from imutils.video import FPS

import argparse
import json
import time

import warnings
warnings.filterwarnings("ignore")

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
Run Inference
'''
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
 
    # Run inference
    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)
     
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop("num_detections"))
    detections = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    detections["num_detections"] = num_detections
 
    # detection_classes should be ints.
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    
    '''
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detections['detection_masks'], detections['detection_boxes'],
            image.shape[0], image.shape[1])      
    
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    detections['detection_masks_reframed'] = detection_masks_reframed.numpy()
    '''
    
    return detections

'''
Show Inference
'''
def show_inference(model, category_index, frame):
    #take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
    
    # Actual detection.
    detections = run_inference_for_single_image(model, image_np)
    
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                       detections["detection_boxes"],
                                                       detections["detection_classes"],
                                                       detections["detection_scores"],
                                                       category_index,
                                                       #instance_masks=output_dict.get('detection_masks_reframed', None),
                                                       use_normalized_coordinates=True,
                                                       min_score_thresh=conf["threshold"],
                                                       line_thickness=5)
    return(image_np)



'''
Load model 
'''
label_dir = pathlib.Path(conf["label"])
category_index = label_map_util.create_category_index_from_labelmap(label_dir, use_display_name=True)

model_dir = pathlib.Path(conf["model"])
start_time = time.time()
detection_model = tf.saved_model.load(str(model_dir))
end_time = time.time()
elapsed_time = end_time - start_time
print()
print("[INFO] model loaded ... took {} seconds".format(elapsed_time))

'''
Image
'''
# load input image
image = cv2.imread(conf["image_input"])
# get spatial dimensions from input image
h, w = image.shape[:2]
print()
print("[INFO] image loaded from file ...")

start_time = time.time()
frame = show_inference(detection_model, category_index, image)
end_time = time.time()
elapsed_time = end_time - start_time
print()
print("[INFO] object(s) detection took {} seconds".format(elapsed_time))

#cv2.namedWindow("SSD Real Time Detections", cv2.WINDOW_NORMAL)
#cv2.imshow("SSD Real Time Detections", image)    
cv2.imshow("SSD Real Time Detection", cv2.resize(frame, (800,600)))
# write new image with detections
cv2.imwrite(conf["image_output"], frame)
# wait for any key being pressed
cv2.waitKey(0)
# destroy opened window
cv2.destroyWindow('Detections') 