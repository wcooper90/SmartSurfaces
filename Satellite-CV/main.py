import cv2
import pandas
import numpy as np
import ee
import os
import random, sys
from inputs.image import LandChunk
from UserInputs import UserInputs
from PIL import Image
from outputs.dataframe import DF


# crop initial images from Google Earth to make sure they are all the same size
def crop_images():
    for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):
        im = Image.open(UserInputs.RAW_IMG_PATH + file)
        width, height = im.size
        assert(width >= UserInputs.DEFAULT_WIDTH and height >= UserInputs.DEFAULT_HEIGHT)

        left = 0
        top = 0
        right = UserInputs.DEFAULT_WIDTH
        bottom = UserInputs.DEFAULT_HEIGHT

        im1 = im.crop((left, top, right, bottom))
        im1.save(UserInputs.CROPPED_IMG_PATH + file)

# find the green in all images
def find_green():
    for i, file in enumerate(os.listdir(UserInputs.CROPPED_IMG_PATH)):
        im = cv2.imread(UserInputs.CROPPED_IMG_PATH + file)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        # can be changed depending on the environment
        mask = cv2.inRange(hsv, (20, 10, 10), (100, 255,255))

        ## slice the green
        imask = mask>0
        green = np.zeros_like(im, np.uint8)
        green[imask] = im[imask]

        ## save
        cv2.imwrite(UserInputs.GREEN_IMG_PATH + file, green)


# find the contours in all images (to be used for finding roofs later)
def find_contours():
    for i, file in enumerate(os.listdir(UserInputs.GREEN_IMG_PATH)):
        im = cv2.imread(UserInputs.GREEN_IMG_PATH + file)

        # Grayscale
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Finding Contours
        _, thresh = cv2.threshold(imgray, 120, 120, 140)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        screeCnt = 0
        for c in contours:
            # approximate the contour
            epsilon = 0.08 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            screeCnt = approx

        # a white backdrop the same size as the rest of video
        white = np.full((UserInputs.DEFAULT_HEIGHT, UserInputs.DEFAULT_HEIGHT), 255, dtype=np.uint8)
        # Draw all contours
        # -1 signifies drawing all contours
        # img = cv2.drawContours(white, contours, -1, (10, 0, 40), int((counter/2)) % 3 + 1)
        img = cv2.drawContours(white, contours, -1, (10, 0, 40), 2)
        cv2.imwrite(UserInputs.CONTOURS_IMG_PATH + file, img)


# calculate the percentage of green pixels in individual images
def percent_green():
    for i, file in enumerate(os.listdir(UserInputs.GREEN_IMG_PATH)):
        im = np.asarray(Image.open(UserInputs.GREEN_IMG_PATH + file))

        num_pixels = UserInputs.DEFAULT_WIDTH * UserInputs.DEFAULT_HEIGHT
        green_pixels = 0
        for line in im:
            for pixel in line:
                comparison = pixel == [0, 0, 0]
                if not (comparison.all()):
                    green_pixels += 1

        print(num_pixels)
        print(green_pixels)
        green_percentage = green_pixels / num_pixels  * 100
        print("Image " + file + " is %" + str(round(green_percentage, 2)) + " green. ")


# find the and create images with only the roofs of images
def find_roofs():

    for i, file in enumerate(os.listdir(UserInputs.CROPPED_IMG_PATH)):
        im = cv2.imread(UserInputs.CROPPED_IMG_PATH + file)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        # can be changed depending on the environment
        mask = cv2.inRange(hsv, (0, 5, 100), (179, 50, 255))

        ## slice the green
        imask = mask>0
        gray = np.zeros_like(im, np.uint8)
        gray[imask] = im[imask]

        ## save
        cv2.imwrite(UserInputs.GRAY_IMG_PATH + file, gray)


# calculate the albedo of an image (LANDSAT strategy)
def albedo():

    return 0


# main function
if __name__ == '__main__':
    # crop_images()
    print("____________IMAGES INITIALIZED____________")
    # find_green()
    print("____________GREEN IMAGES FOUND____________")
    # find_roofs()
    print("_____________ROOF AREAS FOUND_____________")
    # find_contours()
    print("_____________CONTOURS CAPTURED____________")
    # percent_green()
