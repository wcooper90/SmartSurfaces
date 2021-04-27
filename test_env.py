from src.objects.image_curator import Image_Curator
from src.objects.new_city import City
from src.misc_functions import calculate_area
from shapely import geometry

import sys
import os
sys.path.append(os.path.abspath('../../'))
from UserInputs import UserInputs
from api_keys import API_KEYS
from math import log, exp, tan, atan, pi, ceil
import time
import urllib
import urllib.parse
import urllib.request
from src.objects.image import Satellite_Image




boundary = [[-76.7112977,39.3719562],[-76.7112726,39.3543824],[-76.7112337,39.3271971],[-76.7111829,39.2916607],[-76.7111659,39.2778381],[-76.6888115,39.2680858],[-76.6597751,39.2554169],[-76.6178331,39.2371116],[-76.6116152,39.2343976],[-76.5836752,39.2081197],[-76.582525,39.2077508],[-76.5805555,39.2071192],[-76.5497285,39.1972328],[-76.5464898,39.1992526],[-76.5461562,39.1994606],[-76.5298609,39.2096221],[-76.5298503,39.2189088],[-76.5298259,39.2402128],
                [-76.5297584,39.2991345],[-76.5297583,39.2992049],[-76.5297304,39.3239064],[-76.5297299,39.3243919],[-76.5297172,39.3347547],[-76.529677,39.3708243],[-76.5296761,39.3716015],[-76.5296757,39.3719713],[-76.569831,39.37204],[-76.5820693,39.3719669],[-76.6526604,39.3719623],[-76.6529027,39.371961],[-76.6529981,39.371961],[-76.7112977,39.3719562]]
for value in boundary:
    temp = value[0]
    value[0] = value[1]
    value[1] = temp

poly = geometry.Polygon(boundary)
images = []

EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi *  EARTH_RADIUS
INITIAL_RESOLUTION =  EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT =  EQUATOR_CIRCUMFERENCE / 2.0


# https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser
def latlontopixels(lat, lon, zoom):
    mx = (lon *  ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my *  ORIGIN_SHIFT) /180.0
    res =  INITIAL_RESOLUTION / (2**zoom)
    px = (mx +  ORIGIN_SHIFT) / res
    py = (my +  ORIGIN_SHIFT) / res
    return py, px


# https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser
def pixelstolatlon(py, px, zoom):
    res =  INITIAL_RESOLUTION / (2**zoom)
    mx = px * res -  ORIGIN_SHIFT
    my = py * res -  ORIGIN_SHIFT
    lat = (my /  ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx /  ORIGIN_SHIFT) * 180.0
    return round(lon, 7),  round(lat, 7)


def download_helper(x_start, y_start, poly):
    minx, miny, maxx, maxy = poly.bounds
    bounds = [minx, miny, maxx, maxy]
    print(bounds)

    zoom = UserInputs.DEFAULT_ZOOM
    api = API_KEYS()
    key = api.maps_static_api

    # pixel math mostly from https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser
    upper_left_lat, upper_left_lon = float(bounds[2]), float(bounds[1])
    lower_right_lat, lower_right_lon = float(bounds[0]), float(bounds[3])

    # Set some important parameters
    scale = 1
    maxsize = 640

    # convert all these coordinates to pixels
    upper_left_y, upper_left_x =  latlontopixels(upper_left_lat, upper_left_lon, int(zoom))
    lower_right_y, lower_right_x =  latlontopixels(lower_right_lat, lower_right_lon, int(zoom))
    print(upper_left_lat, upper_left_lon)
    print(pixelstolatlon(upper_left_y, upper_left_x, int(zoom)))
    print(lower_right_lat, lower_right_lon)
    print(pixelstolatlon(lower_right_y, lower_right_x, int(zoom)))

    # calculate total pixel dimensions of final image
    x_pixels, y_pixels = lower_right_x - upper_left_x, upper_left_y - lower_right_y

    # calculate rows and columns
    cols, rows = abs(int(ceil(x_pixels/maxsize))), abs(int(ceil(y_pixels/maxsize)))

    print(cols, rows)

    # calculate pixel dimensions of each small image
    largura = UserInputs.DEFAULT_HEIGHT
    altura = UserInputs.DEFAULT_WIDTH

    image_path = UserInputs.CITY_PATH +  "Baltimore" + '/'

    image_counter = 0

    cutoff = False
    for x in range(x_start, cols):
        if image_counter >= UserInputs.DEFAULT_BATCH_SIZE:
            break

        for y in range(y_start, rows):
            print(y)
            if image_counter >= UserInputs.DEFAULT_BATCH_SIZE:
                cutoff = True
                break

            # make more accurate
            dxn = largura * (x)
            dyn = altura * (y)
            latn, lonn =  pixelstolatlon(upper_left_y - dyn, upper_left_x + dxn, int(zoom))

            point = geometry.Point(lonn, latn)
            position = ','.join((str(lonn), str(latn)))
            if not poly.contains(point):
                if y % 7 == 0:
                    print(position + ' not contained in city boundaries ')
                continue

            print (x, y, position)
            urlparams = urllib.parse.urlencode({'center': position,
                                                 'zoom': str(zoom),
                                                 'size': '%dx%d' % (UserInputs.DEFAULT_HEIGHT, UserInputs.DEFAULT_WIDTH),
                                                 'maptype': 'satellite',
                                                 'sensor': 'false',
                                                 'scale': scale,
                                                 'key': key})

            url = 'http://maps.google.com/maps/api/staticmap?' + urlparams

            try:
                # Get the high resolution image
                ipath = image_path + str(image_counter) + ".PNG"
                img = urllib.request.urlretrieve(url, ipath)
                image_object = Satellite_Image(ipath, latn, lonn)
                images.append(image_object)
                image_counter += 1
                y_start += 1
            except IOError:
                print("Couldn't retrieve the image!")


        if cutoff:
            break

        y_start = 0
        x_start += 1
        time.sleep(3)
        print('br')


download_helper(1, 20, poly)
