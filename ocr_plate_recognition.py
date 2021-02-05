##########################################################################################################
#                                                                                                        #
#   Automatic Plate Recognition with Tesseract OCR                                                       #
#   file:ocr_plate_recognition.py                                                                        #
#                                                                                                        #
#   Author: Javier Goya Pérez                                                                            #
#   Date: January 2021                                                                                   #
#                                                                                                        #
##########################################################################################################

import cv2
import numpy as np

# tesseract ocr
# https://github.com/tesserimport pytesseract
import pytesseract
import re

'''
Resize
'''
def resize_image(image, fx, fy, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(image, None, fx, fy, interpolation)

'''
Grayscale
'''
def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

'''
Remove noise
'''
def remove_noise(image, kernel_size=5):
    #return cv2.GaussianBlur(image, (kernel_size,kernel_size), 0)
    return cv2.medianBlur(image, kernel_size)

'''
Threshold
'''
def binary_thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

'''
Erosion
'''
def erosion_morph(image, kernel, iterations=1):
    return cv2.erode(image, kernel, iterations = 1)

'''
Dilation
'''
def dilation_morph(image, kernel, iterations=1):
    return cv2.dilate(image, kernel, iterations = 1)

'''
Opening
'''
def opening_morph(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

'''
Closing
'''
def closing_morph(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

'''
Contours
'''
def get_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    return sorted_contours

'''
Tesseract options
'''
def tesseract_options(psm, oem):
    # Page Segmentation Modes
    # OCR Engine modes
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)
    options += "--oem {}".format(oem)
    return options

'''
Recognize plate
'''
def recognize_plate(image):
    # create blank string to hold license plate number
    plate_num = ""

    # resize image to three times as large as original for better readability
    image_resized = resize_image(image, 3, 3)
    image_h, image_w = image_resized.shape[:2]
    #cv2.imwrite("media/output/matricula_resized.jpg", image_resized)
    
    # convert to grayscale
    gray = convert_grayscale(image_resized)
    #cv2.imwrite("media/output/matricula_gray.jpg", gray)
    
    # remove noise
    blurred = remove_noise(gray)
    #cv2.imwrite("media/output/matricula_blurred.jpg", blurred)

    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = binary_thresholding(blurred)    
    #cv2.imwrite("media/output/matricula_thresh.jpg", thresh)
    
    # create rectangular kernel for morphological operations
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    dilation = dilation_morph(thresh, rect_kern)
    #cv2.imwrite("media/output/matricula_dilation.jpg", dilation)

    # find contours of regions of interest within license plate
    contours = get_contours(dilation)

    i = 0
    # loop through contours and find individual letters and numbers in license plate
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        
        # avoid beginning of the plate (country id)
        if x < int(image_w/25):
            continue
    
        # filter by width and height of the contour
        if w> image_w/2 or w*h < 100 or image_h/h > 2.5 or h/image_h > 0.7:
            continue

        # draw the rectangle
        rect = cv2.rectangle(gray, (x,y), (x+w, y+h), (0,255,0),2)
        #cv2.imwrite("media/output/matricula_contours.jpg", gray)
        
        # region of interest
        roi = dilation[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        
        i = i+1
        # show each roi as an img
        #roi_filename = "media/output/roi"+str(i)+".jpg"
        #cv2.imwrite(roi_filename, roi)
        
        # tesseract character recognition
        options = tesseract_options(psm=8, oem=3)
        try:
            text = pytesseract.image_to_string(roi, config=options)
        
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except:
            text = None
    
    if plate_num == "":
        # tesseract character recognition
        options = tesseract_options(psm=7, oem=3)
        try:
            text = pytesseract.image_to_string(opening, config=options)

            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except:
            text = None

    if len(plate_num) >= 6:
        return(plate_num)
    else:
        return ("")