import sys
import os
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs
from .create_images import mkdir, create_images, random_areas
import cv2
import pandas
import numpy as np
import random
import tqdm
import time
from PIL import Image, ImageFilter
from src.shapedetector import ShapeDetector


class City():
    """
    City is a class which finds and stores images for a specific city (in the US),
    and has functions which can calculate albedo, percentage green/canopy, and
    sloped/nonsloped roofs using computer vision.
    Main functions:
    percent_green; percent_trees; calculate_albedo; calculate_HSroofs;
    calculate_LSroofs.
    Supporting functions:
    __init__; find_green; crop_images; find_roofs; find_contours

    This class' local variables will eventually be used in the final dataframe,
    which can be done through main.py.

    """

    def __init__(self, name, coords, num_images):

        self.name = name
        self.coords = coords
        self.num_images = num_images
        self.contoured = None
        self.contours = 0
        self.albedo = 0
        self.trees = 0
        self.LSroofs = 0
        self.HSroofs = 0
        self.percentGreen = 0

        # make sure path is created
        path = mkdir(self.name)
        self.images_path = path



    def find_raw_images(self, new_images=True):
        maxX = self.coords[0] + UserInputs.CITY_MARGINS
        maxY = self.coords[1] + UserInputs.CITY_MARGINS
        minX = self.coords[0] - UserInputs.CITY_MARGINS
        minY = self.coords[1] - UserInputs.CITY_MARGINS

        if new_images:

            areas = random_areas(maxX, maxY, minX, minY, self.num_images)

            assert(self.images_path is not None)

            create_images(areas, self.images_path)


        self.mount_images(self.images_path, UserInputs.RAW_IMG_PATH)



    def mount_images(self, src_path, dest_path):

        for file in os.listdir(src_path):

            im = Image.open(src_path + file)
            im.save(dest_path + file)

        print("_________"+ str(self.name) + " IMAGES MOUNTED_________")



    # calculate the albedo of an image (LANDSAT strategy)
    def calculate_albedo(self):
        return 0


    # crop initial images from Google Earth to make sure they are all the same size
    def crop_images(self):
        # for i, file in enumerate(os.listdir(self.images_path)):
        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):

            # im = Image.open(self.images_path + file)
            im = Image.open(UserInputs.RAW_IMG_PATH + file)

            width, height = im.size
            assert(width >= UserInputs.DEFAULT_WIDTH and height >= UserInputs.DEFAULT_HEIGHT)

            left = 0
            top = 0
            right = UserInputs.DEFAULT_WIDTH
            bottom = UserInputs.DEFAULT_HEIGHT

            im1 = im.crop((left, top, right, bottom))
            im1.save(UserInputs.CROPPED_IMG_PATH + file)
            im1.save(UserInputs.ALTERED_IMG_PATH + file)
        print("____________IMAGES INITIALIZED____________")


    # find the green in all images
    def find_greenery(self):
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
        print("____________GREEN IMAGES FOUND____________")


    # calculate the percentage of green pixels in individual images
    def percent_green(self):
        percentages = []
        counter = 0
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
            counter += 1
            percentages.append(green_percentage)

        self.percentGreen = sum(percentages) / counter

    def calculate_trees(self):
        return 0

    def calculate_LSroofs(self):
        return 0

    def calculate_HSroofs(self):
        return 0


    def remove_color(self, low_threshold, high_threshold):
        for i, file in enumerate(os.listdir(UserInputs.ALTERED_IMG_PATH)):
            im = cv2.imread(UserInputs.ALTERED_IMG_PATH + file)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            ## mask of green (36,25,25) ~ (86, 255,255)
            # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
            # can be changed depending on the environment
            mask = cv2.inRange(hsv, low_threshold, high_threshold)

            ## slice the green
            imask = mask>0
            color = np.zeros_like(im, np.uint8)
            color[imask] = im[imask]

            ## save
            cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, im - color)
        print("__________COLOR REMOVAL COMPLETE__________")


    # find the and create images with only the roofs of images
    def find_roofs(self):

        roof_counter = 0

        for i, file in enumerate(os.listdir(UserInputs.CROPPED_IMG_PATH)):

            # sharpen image
            imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)
            imageObject = imageObject.filter(ImageFilter.SHARPEN)
            imageObject.save(UserInputs.ALTERED_IMG_PATH + file)

            # contrast for image
            img = cv2.imread(UserInputs.ALTERED_IMG_PATH + file)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)

            #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv2.merge((cl,a,b))

            #-----Converting image from LAB Color model to RGB model--------------------
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, final)

            im = cv2.imread(UserInputs.ALTERED_IMG_PATH + file, cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(UserInputs.CROPPED_IMG_PATH + file)
            _,threshold = cv2.threshold(im, 110, 255,
                            cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:


                rect = cv2.boundingRect(cnt)
                if rect[2] > 200 or rect[3] > 200 or rect[2] < 30 or rect[3] < 30:
                    continue

                x,y,w,h = rect


                roi = im2[y:y+h, x:x+w]
                cv2.imwrite(UserInputs.ROOFS_IMG_PATH + str(roof_counter) + ".jpg", roi)
                roof_counter += 1

                area = cv2.contourArea(cnt)

                # Shortlisting the regions based on there area.
                if area > 800 and area < 15000:
                    approx = cv2.approxPolyDP(cnt,
                                              0.008 * cv2.arcLength(cnt, True), True)

                    # Checking if the no. of sides of the selected region is 7.
                    # if(len(approx) >= 4):
                    cv2.drawContours(im2, [approx], 0, (40, 10, 255), 3)

                cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)

            ## save
            cv2.imwrite(UserInputs.GRAY_IMG_PATH + file, im2)

        print("_____________ROOF AREAS FOUND_____________")


    # find the contours in all images (to be used for finding roofs later)
    def find_contours(self):
        for i, file in enumerate(os.listdir(UserInputs.GREEN_IMG_PATH)):
            im = cv2.imread(UserInputs.GRAY_IMG_PATH + file)

            # Grayscale
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # Finding Contours
            _, thresh = cv2.threshold(imgray, 120, 120, 140)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
        print("_____________CONTOURS CAPTURED____________")
