import sys
import os
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs
from .create_images import mkdir, create_images, random_areas
from .misc_functions import sharp, calculate_area
import matplotlib.pyplot as plt
import cv2
import pandas
import numpy as np
import random
import scipy.ndimage
from deepforest import deepforest
from deepforest import get_data
import tqdm
import time
from src.shapedetector import ShapeDetector
import PIL
from PIL import Image, ImageFilter, ImageFile, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True


class City():
    """
    City is a class which finds and stores images for a specific city (in the US),
    and has functions which can calculate albedo, percentage green/canopy, and
    sloped/nonsloped roofs using computer vision.

    Main functions:
    percent_green; percent_trees; calculate_albedo; calculate_roofs

    Supporting functions:
    __init__; find_green; crop_images; find_roofs; find_contours

    Logistical functions:
    integrate; find_raw_images

    This class' local variables will eventually be used in the final dataframe,
    which can be done through main.py.

    """

    def __init__(self, name, coords, num_images, df, column):

        self.name = name
        self.coords = coords
        self.batch_size = num_images
        self.images_covered_b = 0
        self.images_brightness = 0
        self.contoured = None
        self.contours = 0
        self.albedo = 0
        self.albedo_calculations = 0
        self.treeArea = 0
        self.treePixels = 0
        self.percentTrees = None
        self.roofCounter = 0
        self.roofArea = 0
        self.roofPixels = 0
        self.percentGreen = None
        self.areaCovered = 0
        self.percentAreaCovered = 0
        self.column = column
        self.latitude = float(df["Location"][self.column][0])
        self.longitude = float(df["Location"][self.column][1])
        self.feetPerPixel = calculate_area(self.latitude)
        self.tileArea = self.feetPerPixel * UserInputs.DEFAULT_HEIGHT * UserInputs.DEFAULT_WIDTH

        try:
            self.area = float(df["Area (mi^2)"][self.column])
            if not self.area:
                print("Area has not been found for city of " + self.name)
            else:
                self.area *= UserInputs.SMILES_TO_SFEET
                # print("Area of " + self.name + " is " + str(self.area) + " square feet.")

        except:
            print("City of " + self.name + " is not in the dataframe! Please add it before declaring this object. ")

        # make sure path is created
        path = mkdir(self.name)
        self.images_path = path


    def find_raw_images(self, num, new_images=True, replacement=False, file=None):
        maxX = self.coords[0] + UserInputs.CITY_MARGINS
        maxY = self.coords[1] + UserInputs.CITY_MARGINS
        minX = self.coords[0] - UserInputs.CITY_MARGINS
        minY = self.coords[1] - UserInputs.CITY_MARGINS

        if new_images:

            areas = random_areas(maxX, maxY, minX, minY, num)

            assert(self.images_path is not None)

            image_nums = []
            if not replacement:
                image_nums = np.arange(0, num)
            else:
                image_nums = [int("".join([char for char in file if char.isdigit()]))]

            create_images(areas, image_nums, self.images_path)

        self.mount_images(self.images_path, UserInputs.RAW_IMG_PATH)
        self.mount_images(self.images_path, UserInputs.ALTERED_IMG_PATH)


        if not replacement:
            self.areaCovered += self.batch_size * self.tileArea
            self.percentAreaCovered = self.areaCovered / self.area * 100


    def mount_images(self, src_path, dest_path):

        for file in os.listdir(src_path):

            im = Image.open(src_path + file)
            im.save(dest_path + file)

        print("_________"+ str(self.name.upper()) + " IMAGES MOUNTED__________")



    # calculate the albedo of an image (LANDSAT strategy)
    def calculate_albedo(self):

        albedo_sum = 0
        num = len(os.listdir(UserInputs.RAW_IMG_PATH))

        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):

            brightness = self.brightness(UserInputs.RAW_IMG_PATH + file)
            self.images_brightness += brightness
            self.images_covered_b += 1

            brightest_pixel = self.brightest(UserInputs.RAW_IMG_PATH + file)

            if brightest_pixel > 240:
                albedo_sum += brightness / brightest_pixel * 0.65
            else:
                difference = 240 - brightest_pixel
                standard_albedo = 0.65 - 0.01 * difference
                if standard_albedo < 0.15:
                    standard_albedo =  0.15
                albedo_sum += brightness / brightest_pixel * standard_albedo

        albedo = albedo_sum / num
        self.albedo = (self.albedo_calculations * self.albedo + albedo) / (self.albedo_calculations + 1)

        print("_____________ALBEDO CALCULATED____________")


    def brightness(self, im_file):
        im = Image.open(im_file).convert('L')
        stat = PIL.ImageStat.Stat(im)
        return stat.mean[0]

    # manually search for brightest pixel, stop if find something above 240
    def brightest(self, im_file):
        res = np.array(cv2.imread(im_file, cv2.IMREAD_GRAYSCALE))
        return max(map(max, res))


    # deprecated
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
        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):
            im = cv2.imread(UserInputs.RAW_IMG_PATH + file)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            ## mask of green (36,25,25) ~ (86, 255,255)
            # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
            # can be changed depending on the environment
            mask = cv2.inRange(hsv, UserInputs.LOW_GREEN, UserInputs.HIGH_GREEN)

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

            green_percentage = green_pixels / num_pixels  * 100
            # print("Image " + file + " is %" + str(round(green_percentage, 2)) + " green. ")
            counter += 1
            percentages.append(green_percentage)

        self.percentGreen = sum(percentages) / counter

        print('______' + str(self.name) + ' is ' + str(round(self.percentGreen, 5)) + '% green______')


    def calculate_trees(self):

        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):

            test_model = deepforest.deepforest()
            test_model.use_release()

            image_path = get_data(UserInputs.GREEN_IMG_PATH + file)

            # use this prediction to save images instead of calculating area
            boxes1 = test_model.predict_image(image_path=image_path)
            # use this prediction to calculate area
            boxes2 = test_model.predict_image(image_path=image_path, show=False, return_plot=False)

            # for tiles, if can implement
            # Window size of 300px with an overlap of 25% among windows for this small tile.
            # raster_path = get_data(UserInputs.RAW_IMG_PATH + file)
            # predicted_raster = test_model.predict_tile(raster_path, return_plot = True, patch_size=300,patch_overlap=0.25)


            for box in boxes2.iterrows():
                x = float(box[1]['xmax']) - float(box[1]['xmin'])
                y = float(box[1]['ymax']) - float(box[1]['ymin'])
                self.treePixels += x * y

            # Show image, matplotlib expects RGB channel order, but keras-retinanet predicts in BGR
            plt.imshow(boxes1[...,::-1])
            plt.show()

            plt.savefig(UserInputs.TREES_IMG_PATH + file)


        self.treeArea = self.treePixels * self.feetPerPixel
        self.percentTrees = self.treeArea / self.areaCovered * 100
        print(self.treeArea)
        print(self.areaCovered)
        print("_____________TREE AREAS FOUND_____________")


    def calculate_roofs(self):
        for i, file in enumerate(os.listdir(UserInputs.FINAL_ROOFS_IMG_PATH)):
            im = cv2.imread(UserInputs.FINAL_ROOFS_IMG_PATH + file, cv2.IMREAD_GRAYSCALE)
            self.roofPixels += cv2.countNonZero(im)

        roofArea = self.roofPixels * self.feetPerPixel
        proportion = 100 / self.percentAreaCovered
        self.roofArea = roofArea * proportion / UserInputs.SMILES_TO_SFEET


    def remove_color(self, low_threshold, high_threshold, path=UserInputs.ALTERED_IMG_PATH):
        for i, file in enumerate(os.listdir(path)):
            im = cv2.imread(path + file)
            new_image = self.remove_color_helper(low_threshold, high_threshold, path + file)

            # if the greened image is the same as original, find a new raw image
            while np.array_equal(new_image, im):
                print('helo')
                self.find_raw_images(1, replacement=True, file=file)
                new_image = self.remove_color_helper(low_threshold, high_threshold, path + file)

            ## save
            cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, new_image)
        print("__________COLOR REMOVAL COMPLETE__________")


    def remove_color_helper(self, low_threshold, high_threshold, file):
        im = cv2.imread(file)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        # can be changed depending on the environment
        mask = cv2.inRange(hsv, low_threshold, high_threshold)
        ## slice the green
        imask = mask>0
        color = np.zeros_like(im, np.uint8)
        color[imask] = im[imask]

        new_image = im - color

        return new_image



    def alter_images(self, sharpen=True, contrast=True, brighten=True, grayscale=False):
        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):

            if contrast:
                imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)

                enhancer = ImageEnhance.Contrast(imageObject)

                factor = 1.2 #increase contrast
                im_output = enhancer.enhance(factor)
                im_output.save('more-contrast-image.png')
                im_output.save(UserInputs.ALTERED_IMG_PATH + file)

            # sharpen image
            if sharpen:
                imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)
                enhancer = ImageEnhance.Sharpness(imageObject)
                factor = 2
                im_output = enhancer.enhance(factor)
                im_output.save(UserInputs.ALTERED_IMG_PATH + file)
                # img = cv2.imread(UserInputs.ALTERED_IMG_PATH + file)
                # img = sharp(img)
                # cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, img)

            if brighten:
                imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)
                enhancer = ImageEnhance.Brightness(imageObject)
                enhanced_im = enhancer.enhance(UserInputs.BRIGHTNESS_INCREASE)
                enhanced_im.save(UserInputs.ALTERED_IMG_PATH + file)

            if grayscale:
                img = cv2.imread(UserInputs.ALTERED_IMG_PATH + file)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, gray)


    # find the and create images with only the roofs of images
    def find_roofs(self):

        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):


            im = cv2.imread(UserInputs.ALTERED_IMG_PATH + file, cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(UserInputs.RAW_IMG_PATH + file)
            _,threshold = cv2.threshold(im, 127, 255,
                            cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_SIMPLE)

            rects = []

            for cnt in contours:


                rect = cv2.boundingRect(cnt)

                if rect[2] > 250 or rect[3] > 250 or rect[2] < 30 or rect[3] < 30:
                    continue
                elif rect[2] > 3 * rect[3] or rect[3] > 3 * rect[2]:
                    continue

                rects.append(rect)

                x,y,w,h = rect

                area = cv2.contourArea(cnt)

                # Shortlisting the regions based on there area.
                if area > 800 and area < 15000:
                    approx = cv2.approxPolyDP(cnt,
                                              0.008 * cv2.arcLength(cnt, True), True)

                    # Checking if the no. of sides of the selected region is 7.
                    # if(len(approx) >= 4):
                    # cv2.drawContours(im2, [approx], 0, (40, 10, 255), 3)

                cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)

            im3 = cv2.imread(UserInputs.RAW_IMG_PATH + file)
            for rect in rects:
                x, y, w, h = rect
                roi1 = im[y:y+h, x:x+w]
                roi2 = im3[y:y+h, x:x+w]
                cv2.imwrite(UserInputs.ROOFS_IMG_PATH + str(self.roofCounter) + ".PNG", roi2)
                blur = cv2.GaussianBlur(roi1,(5,5),0)
                _, fix  = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite(UserInputs.FINAL_ROOFS_IMG_PATH + str(self.roofCounter) + ".PNG", fix)
                self.roofCounter += 1

            ## save
            cv2.imwrite(UserInputs.GRAY_IMG_PATH + file, im2)

        print("_____________ROOF AREAS FOUND_____________")


    # find the contours in all images (to be used for finding roofs later)
    def find_contours(self):
        for i, file in enumerate(os.listdir(UserInputs.GREEN_IMG_PATH)):
            im = cv2.imread(UserInputs.ALTERED_IMG_PATH + file)

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
            img = np.full((UserInputs.DEFAULT_HEIGHT, UserInputs.DEFAULT_HEIGHT), 255, dtype=np.uint8)
            # Draw all contours
            # -1 signifies drawing all contours
            # img = cv2.drawContours(white, contours, -1, (10, 0, 40), int((counter/2)) % 3 + 1)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200:
                    img = cv2.drawContours(img, cnt, -1, (10, 0, 40), 2)
            cv2.imwrite(UserInputs.CONTOURS_IMG_PATH + file, img)
        print("_____________CONTOURS CAPTURED____________")


    # input data into city dataframe
    def integrate(self, df):
        df['Albedo'][self.column] = self.albedo
        df['Greenery (%)'][self.column] = self.percentGreen
        df['Roofs (mi^2)'][self.column] = self.roofArea
        df['Trees (%)'][self.column] = self.percentTrees
        df['Area Calculated (%)'][self.column] = self.percentAreaCovered

        print("----------------" + self.name.upper() + " DATA INTEGRATED---------------")
