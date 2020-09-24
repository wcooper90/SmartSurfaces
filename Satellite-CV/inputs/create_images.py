import ee
import sys
import os
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs


def random_areas():
    areas = []
    return areas


def write_images(areas):
    print('will take these areas and put the resulting images in the raw images folder')
    return 0


def mkdir(city):
    try:
        os.mkdir(UserInputs.PATH + 'images/cities/' + city)
        return UserInputs.PATH + 'images/cities/' + city
    except:
        print('Path for ' + city + ' already exists!')
        return 'E'
