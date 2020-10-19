import ee
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

def create_images(areas, path):
    # Create a new instance of GoogleMap Downloader

    length = len(areas)

    for i in range(length):

        random.seed(areas[i][1])

        secs = random.randint(0, 3)

        try:
            # Get the high resolution image
            img = urllib.request.urlretrieve("http://maps.googleapis.com/maps/api/staticmap?center=" + str(areas[i][0]) + ","+ str(areas[i][1]) + "&zoom=19&size=640x640&sensor=false&maptype=satellite&key=AIzaSyC4qq5ofJlgc4bvMofyrd0Q6Sk_BPBNKPE&v=3", path + str(i) + ".PNG")

        except IOError:
            print("Could not generate the image!")

        if i != length:
            time.sleep(secs)





def random_areas(maxX, maxY, minX, minY, num):
    # random seed created from coords
    random.seed(maxX * maxY - minX * minY)

    areas = []
    for i in range(num):
        areas.append([round(random.uniform(minX, maxX), UserInputs.ZOOM_DECIMALS), round(random.uniform(minY, maxY), UserInputs.ZOOM_DECIMALS)])
    print("_________CITY IMAGES RANDOMIZED_________")
    return areas



def mkdir(city):
    try:
        os.mkdir(UserInputs.PATH + 'images/cities/' + city)
        return UserInputs.PATH + 'images/cities/' + city + '/'
    except:
        print('Path for ' + city + ' already exists!')
        return UserInputs.PATH + 'images/cities/' + city + '/'
