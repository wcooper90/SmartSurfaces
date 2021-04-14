import sys
import os
sys.path.append(os.path.abspath('../../'))
from UserInputs import UserInputs



class City():
    """

    """

    def __init__(self):
        self.images = []
        pass


    def print_line(self):
        print("-" * 100)


    def find_raw_images(self):
        # image_curator object, feed in geojson file path
        # self.images = image_curator(geojson_path)
        pass



    # control flow for this object, so only have to call once in main
    def control_flow(self):
        pass

    # calculate the albedo of an image (LANDSAT strategy)
    def calculate_albedo(batch_num):
        pass


    # find the green in all images
    def find_greenery(self, batch_num):
        pass


    # calculate the percentage of green pixels in individual images
    def percent_green(self):
        pass


    def find_trees(self):
        pass


    def percent_trees(self):
        pass


    def find_roofs(self):
        pass

    def percent_roofs(self):
        pass


    # is this needed? 
    def find_contours(self):
        pass


    # input data into city dataframe
    def integrate(self, df):
        pass
