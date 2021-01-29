# button.py
from gpiozero import Button
import sys
import subprocess

button = Button(25)

command = "/home/pi/Workspace/.virtualenvs/tfm/bin/python"
filename = "/home/pi/Workspace/custom_detector_tf2/tflite_object_detection_camera.py"
options = "--conf"
json = "/home/pi/Workspace/custom_detector_tf2/conf.json"

def start_video():
    print("[INFO] Button pressed ... starting video")
    
    global proc
    
    args=[]
    args.append(command)
    args.append(filename)
    args.append(options)
    args.append(json)
    
    proc = subprocess.Popen(args)
        
def stop_video():
    if proc != None:
        proc.terminate()
        print("[INFO] Button released ... stopping video")
        print("")
    else:
        print("[INFO] No video running")
        print("")        
        
try:
    while True:
        button.when_pressed = start_video
        button.when_released = stop_video

except KeyboardInterrupt:
    sys.exit(0)