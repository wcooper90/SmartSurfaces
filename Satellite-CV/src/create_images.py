import sys
import os
import urllib.request
from PIL import Image
import os
import math
import random
import tqdm
import time
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs


def create_images(areas, image_nums, path):
    # Create a new instance of GoogleMap Downloader

    assert(len(areas) == len(image_nums))

    length = len(areas)

    for i in range(length):

        random.seed(areas[i][1])
        secs = random.randint(0, 3)

        try:
            # Get the high resolution image
            img = urllib.request.urlretrieve("http://maps.googleapis.com/maps/api/staticmap?center=" + str(areas[i][0]) + ","+ str(areas[i][1]) + "&zoom=" + UserInputs.DEFAULT_ZOOM + "&size=640x640&sensor=false&maptype=satellite&key=YOURAPIKEYHERE&v=3", path + str(image_nums[i]) + ".PNG")

        except IOError:
            print("Could not generate the image!")

        if i != length:
            time.sleep(secs)


def random_areas(maxX, maxY, minX, minY, num):
    # random seed created from coords
    random.seed(maxX * round(random.uniform(minY, maxY), 5))

    areas = []
    for i in range(num):
        areas.append([round(random.uniform(minX, maxX), UserInputs.ZOOM_DECIMALS), round(random.uniform(minY, maxY), UserInputs.ZOOM_DECIMALS)])
    print("__________CITY IMAGES RANDOMIZED__________")
    return areas



def mkdir(city):
    try:
        os.mkdir(UserInputs.PATH + 'images/cities/' + city)
        return UserInputs.PATH + 'images/cities/' + city + '/'
    except:
        print('Path for ' + city + ' already exists!')
        return UserInputs.PATH + 'images/cities/' + city + '/'
