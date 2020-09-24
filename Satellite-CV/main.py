import cv2
import pandas
import numpy as np
import ee
import os
import random, sys
from inputs.city import City
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
    # print("_____________ROOF AREAS FOUND_____________")
    # print("_____________CONTOURS CAPTURED____________")

    all_columns = UserInputs.DEFAULT_COLUMNS + ['albedo', 'HS Roofs', 'LS Roofs', 'Greenery']

    data = DF(all_columns, UserInputs.DEFAULT_COLUMNS, UserInputs.DEFAULT_SCRAPING_URL)

    data.add_city('Boston')
    data.add_city('Cambridge')
    data.add_city('New Haven')
    data.add_city('Houston')
    data.add_city('Dallas')

    data.print_df()


    boston = City('Boston')

    boston.crop_images()
    print("____________IMAGES INITIALIZED____________")

    boston.find_green()
    print("____________GREEN IMAGES FOUND____________")

    boston.percent_green()

    # data.write_excel()
