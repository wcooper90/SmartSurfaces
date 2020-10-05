import cv2
import pandas
import numpy as np
import ee
import os
import random, sys
from src.city import City
from src.create_images import create_images
from UserInputs import UserInputs
from PIL import Image
from outputs.dataframe import DF


# delete all the photos, making room for analysis of new city
def delete_photos():
    for file in os.listdir(UserInputs.CROPPED_IMG_PATH):
        os.remove(UserInputs.CROPPED_IMG_PATH + file)
    for file in os.listdir(UserInputs.GRAY_IMG_PATH):
        os.remove(UserInputs.GRAY_IMG_PATH + file)
    for file in os.listdir(UserInputs.GREEN_IMG_PATH):
        os.remove(UserInputs.GREEN_IMG_PATH + file)
    for file in os.listdir(UserInputs.CONTOURS_IMG_PATH):
        os.remove(UserInputs.CONTOURS_IMG_PATH + file)
    for file in os.listdir(UserInputs.RAW_IMG_PATH):
        os.remove(UserInputs.RAW_IMG_PATH + file)


# main function
if __name__ == '__main__':

    all_columns = UserInputs.DEFAULT_COLUMNS + ['albedo', 'HS Roofs', 'LS Roofs', 'Greenery']

    data = DF(all_columns, UserInputs.DEFAULT_COLUMNS, UserInputs.DEFAULT_SCRAPING_URL)

    # data.add_city_values('Boston')
    # data.add_city_values('Cambridge')
    # data.add_city_values('New Haven')
    # data.add_city_values('Houston')
    # data.add_city_values('Dallas')
    #
    # data.print_df()


    boston = City('Boston', [42.361145, -71.057083], 5)
    # boston.find_raw_images()
    boston.crop_images()
    boston.find_greenery()
    boston.remove_color(UserInputs.LOW_YELLOW, UserInputs.HIGH_YELLOW)
    boston.remove_color(UserInputs.LOW_GREEN, UserInputs.HIGH_GREEN)

    # boston.percent_green()
    boston.find_roofs()
    boston.find_contours()

    # data.write_excel()
