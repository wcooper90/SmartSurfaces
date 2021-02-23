import sys
import os
from UserInputs import UserInputs
import matplotlib.pyplot as plt
import skimage.measure
from skimage.filters.rank import entropy
from skimage.morphology import disk
import urllib.request
import cv2
import pandas
import numpy as np
import random
import scipy.ndimage
import tqdm
import time
import PIL
from PIL import Image, ImageFilter, ImageFile, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

img = urllib.request.urlretrieve("https://maps.googleapis.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=13&size=600x300&maptype=roadmap&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318&markers=color:red%7Clabel:C%7C40.718217,-73.998284&key=AIzaSyCEyvJOHsuuNMciO70NCwZNn5BpjKY1Sg4")

#
# from PIL import Image
# from PIL import ImageStat
# import math
#
# # function to return average brightness of an image
# # Source: https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
#
# def brightness(im_file):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    r,g,b = stat.mean
#    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))   #this is a way of averaging the r g b values to derive "human-visible" brightness
#
# img_path = '0.PNG'
# img_path2 = '1.PNG'
#
#
# imageObject = Image.open(img_path)
# enhancer = ImageEnhance.Contrast(imageObject)
# factor = 1.5 # increase contrast
# im_output = enhancer.enhance(factor)
# im_output.save(img_path)
#
# folder = os.getcwd() + '/test_images/'
# im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# blur = cv2.GaussianBlur(im,(11,11),0)
# _, fix  = cv2.threshold(blur,190,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# blur2 = cv2.GaussianBlur(fix, (7,7),0)
#
# im2 = cv2.imread(img_path)
#
# _,threshold = cv2.threshold(blur2, 127, 255,
#                 cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(threshold, cv2.RETR_CCOMP,
#                 cv2.CHAIN_APPROX_SIMPLE)
#
# rects = []
# treeCounter = 0
# for cnt in contours:
#     rect = cv2.boundingRect(cnt)
#     x,y,w,h = rect
#     rects.append(rect)
#     area = cv2.contourArea(cnt)
#
#     # Shortlisting the regions based on there area.
#     if area > 400 and area < 300000:
#         approx = cv2.approxPolyDP(cnt,
#                                   0.0001 * cv2.arcLength(cnt, True), True)
#         # Checking if the no. of sides of the selected region is 7.
#         if(len(approx) >= 3):
#             # cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,40),2)
#             roi1 = im2[y:y+h, x:x+w]
#             roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
#             _, fix  = cv2.threshold(roi1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             blur2 = cv2.GaussianBlur(fix, (5,5),0)
#             array = np.asarray(blur2)
#             en = np.average(entropy(array, disk(10)))
#             sd = roi1.std()
#
#             if en >= 4.5 and en <= 6.5 and sd < 80 and sd > 50:
#                 # cv2.imwrite(UserInputs.TREES_FINAL_PATH + str(self.treeCounter) + ".PNG", blur2)
#                 cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,40),2)
#
#             print('image entropy ' + str(treeCounter) + ': ' + str(en))
#             print('image contrast ' + str(treeCounter) + ': ' + str(roi1.std()))
#
#             cv2.imwrite(folder + str(treeCounter) + ".PNG", roi1)
#             treeCounter += 1
#
# cv2.imwrite(img_path2, im2)
