import sys
import os
sys.path.append(os.path.abspath('../../'))
from UserInputs import UserInputs
from src.objects.image_curator import Image_Curator



class City():
    """

    """

    def __init__(self, city_name):
        self.city_name = city_name
        self.images = []
        self.image_curator = None
        self.max_image_iterations = None


    def print_line(self):
        print("-" * 100)


    def find_raw_images(self):
        # image_curator object, feed in geojson file path
        if not self.image_curator:
            self.image_curator = Image_Curator(self.city_name, UserInputs.DEFAULT_BATCH_SIZE)
            self.image_curator.control_flow()
            self.max_image_iterations = min(UserInputs.MAX_IMAGES_ITERATIONS, (self.image_curator.cols * self.image_curator.rows / UserInputs.DEFAULT_BATCH_SIZE))
        else:
            self.image_curator.control_flow()
            self.images = self.image_curator.images


        self.print_line()


    # control flow for this object, so only have to call once in main
    def control_flow(self):
        self.find_raw_images()
        for i in range(self.max_image_iterations - 1):
            self.find_raw_images()


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
