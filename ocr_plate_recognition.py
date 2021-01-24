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
def remove_noise(image):
    return cv2.GaussianBlur(image, (5,5), 0)
'''
Threshold
'''
def binary_thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

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
Bitwise
'''
def bitwise_operator(operator, image):
    image_copy = image.copy()
    if operator == "and":
        image_copy = cv2.bitwise_and(image_copy)
    elif operator == "not":
        image_copy = cv2.bitwise_not(image_copy)
        
    return image_copy

'''
Roi
'''
def get_roi(image, x, y, w, h):
    # grab character region of image
    roi = image[y-5:y+h+5, x-5:x+w+5]
    
    # perfrom bitwise not to flip image to black text on white background
    roi = bitwise_operator("not", roi)

    return roi

'''
Tesseract options
'''
def tesseract_options(psm=8, oem=1):
    # Page segmentation modes: 8 = Treat the image as a single word.
    # OCR Engine modes: 1 = Neural nets LSTM engine only.
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)
    options += "--oem {}".format(oem)
    return options

'''
Recognize plate
'''
def recognize_plate(image):
    # resize image to three times as large as original for better readability
    image_resized = resize_image(image, 3, 3)
    #cv2.imwrite("media/output/matricula_resized.jpg", image_resized)
    
    # convert to grayscale
    gray = convert_grayscale(image_resized)
    #cv2.imwrite("media/output/matricula_gray.jpg", gray)
    
    # remove noise
    blurred = remove_noise(gray)
    #cv2.imwrite("media/output/matricula_blurred.jpg", blurred)

    # create rectangular kernel for morphological operations
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    # apply erosion to make regions more clear
    erosion = erosion_morph(blurred, rect_kern, iterations = 1)
    #cv2.imwrite("media/output/matricula_erosion.jpg", erosion)

    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = binary_thresholding(erosion)
    #cv2.imwrite("media/output/matricula_thresh.jpg", thresh)
    
    opening = opening_morph(thresh, rect_kern)
    #cv2.imwrite("media/output/matricula_opening.jpg", opening)

    # find contours of regions of interest within license plate
    contours = get_contours(thresh)
    
    # create blank string to hold license plate number
    plate_num = ""
    
    i = 0
    # loop through contours and find individual letters and numbers in license plate
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        height, width = image_resized.shape[:2]
        
        # filter by height of the contour
        if (h / height) < 0.5 or (h / height) > 0.75:
            continue
        
        i = i+1
        
        # draw the rectangle
        rect = cv2.rectangle(gray, (x,y), (x+w, y+h), (0,255,0),2)
        #cv2.imwrite("media/output/matricula_contours.jpg", gray)
        
        # region of interest
        roi = get_roi(thresh, x, y, w, h)
        
        # show each roi as an img
        #roi_filename = "media/output/roi"+str(i)+".jpg"
        #cv2.imwrite(roi_filename, roi)
        
        # tesseract character recognition
        options = tesseract_options(psm=8, oem=1)
        text = pytesseract.image_to_string(roi, config=options)
        
        # clean tesseract text by removing any unwanted blank spaces
        clean_text = re.sub('[\W_]+', '', text)
        plate_num += clean_text
        
    if plate_num != "":
        print("[INFO] recognize_plate ... plate = {}".format(plate_num))
    
    return(plate_num)