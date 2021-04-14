import sys
import os
sys.path.append(os.path.abspath('../../'))
from UserInputs import UserInputs
from src.create_images import mkdir, create_images, random_areas
from src.misc_functions import sharp, calculate_area
import matplotlib.pyplot as plt
import skimage.measure
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2
import pandas
import numpy as np
import random
import scipy.ndimage
from deepforest import deepforest
from deepforest import get_data
import tqdm
import time
import PIL
from PIL import Image, ImageFilter, ImageFile, ImageEnhance, ImageStat
import math
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

    def __init__(self, name, coords, num_images, df, row):

        self.name = name
        self.coords = coords
        self.batch_size = num_images
        self.contoured = None
        self.contours = 0
        self.albedo = 0
        self.iterations = -1
        self.treeCounter = 0
        self.treeArea = 0
        self.treePixels = 0
        self.percentTrees = None
        self.roofCounter = 0
        self.roofArea = 0
        self.roofPixels = 0
        self.percentGreen = 0
        self.treeCounter = 0
        self.areaCovered = 0
        self.percentAreaCovered = 0
        self.row = row
        self.latitude = float(df["Location"][self.row][0])
        self.longitude = float(df["Location"][self.row][1])
        self.feetPerPixel = calculate_area(self.latitude)
        self.tileArea = self.feetPerPixel * UserInputs.DEFAULT_HEIGHT * UserInputs.DEFAULT_WIDTH


        try:
            self.area = float(df["Area (mi^2)"][self.row])
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


    def print_line(self):
        print("-" * 100)


    def find_raw_images(self, num, new_images=True, replacement=False, file=None):

        self.iterations += 1

        maxX = self.coords[0] + UserInputs.CITY_MARGINS
        maxY = self.coords[1] + UserInputs.CITY_MARGINS
        minX = self.coords[0] - UserInputs.CITY_MARGINS
        minY = self.coords[1] - UserInputs.CITY_MARGINS

        if new_images:

            random = self.iterations * self.batch_size + UserInputs.RANDOM_SEED - 32
            # print("random seed: " + str(random))
            areas = random_areas(maxX, maxY, minX, minY, num, random)
            # for area in areas:
            #     print(area)

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
            brightest_pixel = self.brightest(UserInputs.RAW_IMG_PATH + file)

            # change standard albedo of office paper from 0.65 to 0.57 to
            # account for some images not getting as much light as others
            if brightest_pixel > 253:
                # print(brightness)
                albedo_sum += brightness / brightest_pixel * 0.5
            else:
                difference = 253 - brightest_pixel
                standard_albedo = 0.5 - 0.098 * difference
                # print('lower albedo')
                if standard_albedo < 0.15:
                    standard_albedo =  0.15
                    # print('lowest albedo')
                albedo_sum += brightness / brightest_pixel * standard_albedo
                # print(brightness / brightest_pixel * standard_albedo)

        albedo = albedo_sum / num
        self.albedo = (self.iterations * self.albedo + albedo) / (self.iterations + 1)

        print("_____________ALBEDO CALCULATED____________")


    def brightness(self, im_file):
        im = Image.open(im_file).convert('L')
        stat = PIL.ImageStat.Stat(im)
        return stat.mean[0]


    def brightness2(self, im_file):
        im = Image.open(im_file)
        stat = ImageStat.Stat(im)
        r,g,b = stat.mean[:3]
        return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


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

            if np.array_equal(green, im):
                print('Image discarded')
                self.areaCovered -= self.tileArea
                self.batch_size -= 1
                self.percentAreaCovered = self.areaCovered / self.area * 100

            else:
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

        percent = sum(percentages) / counter
        self.percentGreen = (self.percentGreen * self.iterations  + percent) / (self.iterations + 1)
        print('______' + str(self.name) + ' is ' + str(round(self.percentGreen, 5)) + '% green______')


    def standardize(self):

        # convert to RGBA
        # for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):
        #     im2 = Image.open(UserInputs.RAW_IMG_PATH + file)
        #     im2 = im2.convert("RGBA")
        #     datas = im2.getdata()
        #     newData = []
        #     for item in datas:
        #         if item[0] == 0 and item[1] == 0 and item[2] == 0:
        #             newData.append((0, 0, 0, 0))
        #         else:
        #             newData.append(item)
        #     im2.putdata(newData)
        #     im2.save(UserInputs.GREEN_IMG_PATH + file)

        myList = []
        deltaList = []
        b = 0.0
        num_images = len(os.listdir(UserInputs.RAW_IMG_PATH))
        for file in os.listdir(UserInputs.RAW_IMG_PATH):
            myList.append(self.brightness(UserInputs.RAW_IMG_PATH + file))

        avg_brightness = sum(myList)/num_images

        for i, _ in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):
            deltaList.append(avg_brightness - myList[i])

        for k, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):      # for loop runs from image number 1 thru 20
            img_file = Image.open(UserInputs.RAW_IMG_PATH + file)
            img_file = img_file.convert('RGB')     # converts image to RGB format
            pixels = img_file.load()               # creates the pixel map
            for i in range (img_file.size[0]):
                for j in range (img_file.size[1]):
                    r, g, b = img_file.getpixel((i,j))  # extracts r g b values for the i x j th pixel
                    pixels[i,j] = (r+int(deltaList[k]), g+int(deltaList[k]), b+int(deltaList[k])) # re-creates the image

            img_file.save(UserInputs.ALTERED_IMG_PATH + file)


    def find_trees(self):

        treeCounter = 0
        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):

            im = cv2.imread(UserInputs.GREEN_IMG_PATH + file, cv2.IMREAD_GRAYSCALE)
            blur = cv2.GaussianBlur(im,(11,11),0)
            _, fix  = cv2.threshold(blur,190,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            blur2 = cv2.GaussianBlur(fix, (7,7),0)
            im2 = cv2.imread(UserInputs.RAW_IMG_PATH + file)
            _,threshold = cv2.threshold(blur2, 127, 255,
                            cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_SIMPLE)
            rects = []

            for cnt in contours:
                rect = cv2.boundingRect(cnt)
                if rect[2] > UserInputs.MAX_TREE_LENGTH or rect[3] > UserInputs.MAX_TREE_LENGTH or rect[2] < UserInputs.MIN_TREE_LENGTH or rect[3] < UserInputs.MIN_TREE_LENGTH:
                    continue
                x,y,w,h = rect
                rects.append(rect)
                area = cv2.contourArea(cnt)

                # Shortlisting the regions based on there area.
                if area > 400 and area < 300000:
                    approx = cv2.approxPolyDP(cnt,
                                              0.01 * cv2.arcLength(cnt, True), True)

                    if(len(approx) >= 3):
                        roi1 = im[y:y+h, x:x+w]
                        roi2 = im2[y:y+h, x:x+w]
                        _, fix  = cv2.threshold(roi1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        blur2 = cv2.GaussianBlur(fix, (5,5),0)
                        array = np.asarray(blur2)
                        en = np.average(entropy(array, disk(10)))
                        sd = roi1.std()

                        cv2.imwrite(UserInputs.TREES_RAW_PATH + str(treeCounter) + '.png', roi2)

                        greenPixels = cv2.countNonZero(blur2)

                        if sd >= 26 and sd <= 65 and en > 3.8:
                            # cv2.imwrite(UserInputs.TREES_RAW_PATH + str(self.treeCounter) + ".PNG", blur2)
                            cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,40),2)
                            self.treePixels += greenPixels
                        else:
                            cv2.rectangle(im2,(x,y),(x+w,y+h),(255,0,0),2)

                        # print('image ' + str(treeCounter) + ':' + str(en))
                        # print('image ' + str(treeCounter) + ':' + str(sd))
                        # print('image ' + str(treeCounter) + ':' + str(ratio))

                        # else:
                            # if en >= 4.25 and en <= 6.25 and sd < 57 and sd > 35:
                            #     # cv2.imwrite(UserInputs.TREES_RAW_PATH + str(self.treeCounter) + ".PNG", blur2)
                            #     cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,40),2)
                            #     self.treePixels += greenPixels
                            # else:
                            #     cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),2)
                            #
                            #     print('image ' + str(treeCounter) + ':' + str(en))
                            #     print('image ' + str(treeCounter) + ':' + str(sd))
                            #     print('image ' + str(treeCounter) + ':' + str(ratio))

                        treeCounter += 1

            ## save
            cv2.imwrite(UserInputs.TREES_IMG_PATH + file, im2)

        # self.treeCounter = 0
        self.treeArea = self.treePixels * self.feetPerPixel
        percentTrees = self.treeArea / (self.tileArea * self.batch_size) * 100
        # print(percentTrees)
        try:
            self.percentTrees = (self.percentTrees * self.iterations) + percentTrees / (self.iterations + 1)
        except:
            self.percentTrees = percentTrees

        # print(self.treeArea)
        # print(self.areaCovered)
        print("_____________TREE AREAS FOUND_____________")


    def calculate_trees(self):

        length = len(os.listdir(UserInputs.RAW_IMG_PATH))

        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):

            random.seed(i + UserInputs.RANDOM_SEED)
            secs = random.randint(0, 3)

            im = cv2.imread(UserInputs.RAW_IMG_PATH + file)

            test_model = deepforest.deepforest()

            # sometimes there are too many calls to the model, so image is just discarded
            try:
                test_model.use_release()
            except:
                print('tree image discarded')
                continue

            image_path = get_data(UserInputs.GREEN_IMG_PATH + file)

            # use this prediction to save images instead of calculating area
            boxes1 = test_model.predict_image(image_path=image_path)
            # use this prediction to calculate area
            boxes2 = test_model.predict_image(image_path=image_path, show=False, return_plot=False)

            # for tiles, if can implement
            # Window size of 300px with an overlap of 25% among windows for this small tile.
            # raster_path = get_data(UserInputs.RAW_IMG_PATH + file)
            # predicted_raster = test_model.predict_tile(raster_path, return_plot = True, patch_size=300,patch_overlap=0.25)

            treeCounter = 0
            for box in boxes2.iterrows():

                roi = im[int(box[1]['ymin']):int(box[1]['ymax']), int(box[1]['xmin']):int(box[1]['xmax'])]
                im2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(UserInputs.TREES_RAW_PATH + str(treeCounter) + '.png', im2)

                self.treePixels += cv2.countNonZero(im2)
                # cv2.imwrite(UserInputs.FINAL_TREES_IMG_PATH + str(self.treeCounter) + ".PNG", roi)
                treeCounter += 1

            self.treeCounter += 1

            # Show image, matplotlib expects RGB channel order, but keras-retinanet predicts in BGR
            plt.imshow(boxes1[...,::-1])
            plt.show()

            plt.savefig(UserInputs.TREES_IMG_PATH + file)

            if i != length:
                time.sleep(secs)

            # break

        # for i, file in enumerate(os.listdir(UserInputs.FINAL_TREES_IMG_PATH)):
        #     im = cv2.imread(UserInputs.FINAL_TREES_IMG_PATH + file, cv2.IMREAD_GRAYSCALE)
        #     self.treePixels += cv2.countNonZero(im)

        self.treeArea = self.treePixels * self.feetPerPixel
        try:
            self.percentTrees = self.treeArea / (self.treeCounter * self.tileArea) * 100
        except:
            self.percentTress = None
        # print(self.treeArea)
        # print(self.areaCovered)
        print("_____________TREE AREAS FOUND_____________")


    def calculate_roofs(self):
        print('new iterations')

        for i, file in enumerate(os.listdir(UserInputs.FINAL_ROOFS_IMG_PATH)):
            im = cv2.imread(UserInputs.FINAL_ROOFS_IMG_PATH + file, cv2.IMREAD_GRAYSCALE)
            self.roofPixels += cv2.countNonZero(im)

        roofArea = self.roofPixels * self.feetPerPixel / UserInputs.SMILES_TO_SFEET
        proportion = 100 / self.percentAreaCovered

        self.roofArea = roofArea * proportion
        print(self.percentAreaCovered)
        print(self.roofPixels)
        print(self.roofArea)


    def remove_color(self, low_threshold, high_threshold, path=UserInputs.ALTERED_IMG_PATH):
        for i, file in enumerate(os.listdir(path)):
            im = cv2.imread(path + file)
            new_image = self.remove_color_helper(low_threshold, high_threshold, path + file)

            ## save
            cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, new_image)
        print("__________COLOR REMOVAL COMPLETE__________")


    def remove_color_helper(self, low_threshold, high_threshold, file):
        im = cv2.imread(file)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, low_threshold, high_threshold)
        imask = mask>0
        color = np.zeros_like(im, np.uint8)
        color[imask] = im[imask]

        new_image = im - color

        return new_image


    def alter_images(self, sharpen=True, contrast=True, brighten=True, grayscale=False, otsu=True):

        for i, file in enumerate(os.listdir(UserInputs.RAW_IMG_PATH)):

            # contrast
            if contrast:
                brightness = self.brightness(UserInputs.ALTERED_IMG_PATH + file)
                contrast_increase = 0.00012 * (brightness - 150) ** 2 + 1.3
                # print("contrast_increase = " + str(contrast_increase))
                imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)
                enhancer = ImageEnhance.Contrast(imageObject)
                factor = contrast_increase # increase contrast
                im_output = enhancer.enhance(factor)
                im_output.save(UserInputs.ALTERED_IMG_PATH + file)

            # brighten
            if brighten:
                imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)
                enhancer = ImageEnhance.Brightness(imageObject)
                enhanced_im = enhancer.enhance(UserInputs.BRIGHTNESS_INCREASE)
                enhanced_im.save(UserInputs.ALTERED_IMG_PATH + file)

            # sharpen image
            if sharpen:
                imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)
                enhancer = ImageEnhance.Sharpness(imageObject)
                factor = UserInputs.SHARPNESS_INCREASE
                im_output = enhancer.enhance(factor)
                im_output.save(UserInputs.ALTERED_IMG_PATH + file)

            # brighten
            if brighten:
                imageObject = Image.open(UserInputs.ALTERED_IMG_PATH + file)
                enhancer = ImageEnhance.Brightness(imageObject)
                enhanced_im = enhancer.enhance(UserInputs.BRIGHTNESS_INCREASE)
                enhanced_im.save(UserInputs.ALTERED_IMG_PATH + file)

            # otsu's binarization
            if otsu:
                img = cv2.imread(UserInputs.ALTERED_IMG_PATH + file, cv2.IMREAD_GRAYSCALE)
                blur = cv2.GaussianBlur(img,(5,5),0)
                _, fix  = cv2.threshold(blur,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, fix)

            # grayscale
            if grayscale:
                img = cv2.imread(UserInputs.ALTERED_IMG_PATH + file)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(UserInputs.ALTERED_IMG_PATH + file, gray)


    # find the and create images with only the roofs of images
    def find_roofs(self):

        # remove the existing photos from final roofs path
        for file in os.listdir(UserInputs.FINAL_ROOFS_IMG_PATH):
            os.remove(UserInputs.FINAL_ROOFS_IMG_PATH + file)


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
                if rect[2] > UserInputs.MAX_WIDTH_LENGTH or rect[3] > UserInputs.MAX_WIDTH_LENGTH or rect[2] < UserInputs.MIN_WIDTH_LENGTH or rect[3] < UserInputs.MIN_WIDTH_LENGTH:
                    continue
                elif rect[2] > 3 * rect[3] or rect[3] > 3 * rect[2]:
                    continue

                rects.append(rect)
                # x,y,w,h = rect
                area = cv2.contourArea(cnt)

                # Shortlisting the regions based on there area.
                # if area > 800 and area < 15000:
                #     approx = cv2.approxPolyDP(cnt,
                #                               0.008 * cv2.arcLength(cnt, True), True)

                    # Checking if the no. of sides of the selected region is 7.
                    # if(len(approx) >= 4):
                    # cv2.drawContours(im2, [approx], 0, (40, 10, 255), 3)

            im3 = cv2.imread(UserInputs.RAW_IMG_PATH + file)
            for rect in rects:
                x, y, w, h = rect
                roi1 = im[y:y+h, x:x+w]
                roi2 = im3[y:y+h, x:x+w]
                cv2.imwrite(UserInputs.ROOFS_IMG_PATH + str(self.roofCounter) + ".PNG", roi2)
                blur = cv2.GaussianBlur(roi1,(5,5),0)
                _, fix  = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                blur2 = cv2.GaussianBlur(fix, (3,3),0)
                array = np.asarray(blur2)
                en = np.average(entropy(array, disk(10)))
                if en <= UserInputs.MAX_ENTROPY and en >= UserInputs.MIN_ENTROPY:
                    cv2.imwrite(UserInputs.FINAL_ROOFS_IMG_PATH + str(self.roofCounter) + ".PNG", blur2)
                    cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,40),2)
                else:
                    cv2.rectangle(im2,(x,y),(x+w,y+h),(255,0,0),2)
                    # print('image ' + str(self.roofCounter) + ': ' + str(en))
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
        df['Albedo'][self.row] = self.albedo
        df['Greenery (%)'][self.row] = self.percentGreen
        df['Roofs (mi^2)'][self.row] = self.roofArea
        df['Trees (%)'][self.row] = self.percentTrees
        df['Area Calculated (%)'][self.row] = self.percentAreaCovered

        print("----------------" + self.name.upper() + " DATA INTEGRATED---------------")
