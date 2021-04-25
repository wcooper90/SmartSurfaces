from .boundary import Boundary
from shapely import geometry
import urllib
import urllib.parse
import urllib.request
from math import log, exp, tan, atan, pi, ceil
import sys
import os
import time
sys.path.append(os.path.abspath('../../'))
from UserInputs import UserInputs
from api_keys import API_KEYS
from src.create_images import mkdir
from src.objects.image import Satellite_Image


class Image_Curator():
    # from https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser


    def __init__ (self, city_name, batch_size, x_start=0, y_start=0):
        self.status = "complete"
        self.city_name = city_name
        # 2 dimensional array to represent the batches of images
        self.image_coordinates = []
        self.images = []
        self.x_start = x_start
        self.y_start = y_start
        self.batch_size = batch_size
        self.cols = None
        self.rows = None

        # https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser
        self.EARTH_RADIUS = 6378137
        self.EQUATOR_CIRCUMFERENCE = 2 * pi * self.EARTH_RADIUS
        self.INITIAL_RESOLUTION = self.EQUATOR_CIRCUMFERENCE / 256.0
        self.ORIGIN_SHIFT = self.EQUATOR_CIRCUMFERENCE / 2.0


    # interpret geojson file, find image coordinates
    def download_images(self):
        # eventually name of city to be passed through here
        boundary = Boundary(self.city_name)
        poly = geometry.Polygon(boundary.boundary)
        self.download_helper(self.x_start, self.y_start, poly, self.batch_size)

        """
        - produce bounding points
        - start from top left, find first point, make sure bounding points of
            each image are all contained within poly.bounds
        """


    def download_helper(self, x_start, y_start, poly, batch_size):
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
        upper_left_x, upper_left_y = self.latlontopixels(upper_left_lat, upper_left_lon, int(zoom))
        lower_right_x, lower_right_y = self.latlontopixels(lower_right_lat, lower_right_lon, int(zoom))

        # calculate total pixel dimensions of final image
        x_pixels, y_pixels = lower_right_x - upper_left_x, upper_left_y - lower_right_y

        # calculate rows and columns
        cols, rows = int(ceil(x_pixels/maxsize)), int(ceil(y_pixels/maxsize))
        self.cols = cols
        self.rows = rows

        # calculate pixel dimensions of each small image
        largura = UserInputs.DEFAULT_HEIGHT
        altura = UserInputs.DEFAULT_WIDTH

        image_path = UserInputs.CITY_PATH + self.city_name + '/'

        image_counter = 0
        print("x_start: " + str(x_start))
        print("y_start: " + str(y_start))
        for x in range(x_start, cols):
            if image_counter >= UserInputs.DEFAULT_BATCH_SIZE:
                break

            for y in range(y_start, rows):
                if image_counter >= UserInputs.DEFAULT_BATCH_SIZE:
                    break

                # make more accurate
                dxn = largura * (x) * 4.3
                dyn = altura * (y) * 4.3
                latn, lonn = self.pixelstolatlon(upper_left_x + dxn, upper_left_y - dyn, int(zoom))

                point = geometry.Point(latn, lonn)
                position = ','.join((str(latn), str(lonn)))
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
                    self.images.append(image_object)
                    image_counter += 1
                    self.y_start += 1
                except IOError:
                    print("Couldn't retrieve the image!")

            self.x_start += 1
            self.y_start = 0
            time.sleep(3)
            print('br')


    # https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser
    def latlontopixels(self, lat, lon, zoom):
        mx = (lon * self.ORIGIN_SHIFT) / 180.0
        my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
        my = (my * self.ORIGIN_SHIFT) /180.0
        res = self.INITIAL_RESOLUTION / (2**zoom)
        px = (mx + self.ORIGIN_SHIFT) / res
        py = (my + self.ORIGIN_SHIFT) / res
        return px, py


    # https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser
    def pixelstolatlon(self, px, py, zoom):
        res = self.INITIAL_RESOLUTION / (2**zoom)
        mx = px * res - self.ORIGIN_SHIFT
        my = py * res - self.ORIGIN_SHIFT
        lat = (my / self.ORIGIN_SHIFT) * 180.0
        lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
        lon = (mx / self.ORIGIN_SHIFT) * 180.0
        return lat, lon



    # batch image coordinates
    def batcher(self):
        pass


    # call create_images with the image coordinates
    def curate_images(self):
        pass



    # run through images to produce an average brightness, contrast, classification
    # run through images again to assign standarization values to each image object
    def evaluate_images(self):

        pass


    # call image altering functions
    def standardize_images(self):
        pass


    # control flow for this object, returns image objects in batches
    def control_flow(self):
        self.download_images()
