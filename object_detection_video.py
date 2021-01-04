import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf
import pathlib

from collections import defaultdict

from PIL import Image
from IPython.display import display

#from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from imutils.video import VideoStream
from imutils.video import FPS

import argparse
import json
import time

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
                                                       min_score_thresh=.5,
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
print("[INFO] model loaded ... took {} seconds".format(elapsed_time))

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

    start = time.time()
    image = show_inference(detection_model, category_index, frame)
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
        writer = cv2.VideoWriter(conf["video_output"], fourcc, 4, (image.shape[1], image.shape[0]), True)
        
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write processed current frame to the file
    writer.write(image)

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