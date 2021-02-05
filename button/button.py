# button.py
from gpiozero import Button, RGBLED
from colorzero import Color

#code extracted from
#https://cdn.shopify.com/s/files/1/0176/3274/files/instructions_squid_family.pdf?v=1605191491

import sys
import subprocess

def red_changed(value):
    global red
    red = int(value)
    rgb_led.color = Color(red, green, blue)
def green_changed(value):
    global green
    green = int(value)
    rgb_led.color = Color(red, green, blue)
def blue_changed(value):
    global blue
    blue = int(value)
    rgb_led.color = Color(red, green, blue)

button = Button(25)
rgb_led = RGBLED(18, 23, 24)
red = 0
green = 0
blue = 0

command = "/home/pi/Workspace/.virtualenvs/tfm/bin/python"
filename = "/home/pi/Workspace/custom_detector_tf2/tflite_object_detection_camera.py"
options = "--conf"
json = "/home/pi/Workspace/custom_detector_tf2/conf_tflite.json"

def start_video():
    print("[INFO] Button pressed ... starting video")
    
    global proc
    
    args=[]
    args.append(command)
    args.append(filename)
    args.append(options)
    args.append(json)
        
    proc = subprocess.Popen(args)
    
    green_changed(255)
      
def stop_video():    
    if proc != None:
        proc.terminate()
        print("[INFO] Button released ... stopping video")
        print("")
    else:
        print("[INFO] No video running")
        print("")
    
    green_changed(0)
        
try:
    while True:
        button.when_pressed = start_video
        button.when_released = stop_video

except KeyboardInterrupt:
    sys.exit(0)