import cv2
import pandas
import numpy as np
from tqdm import tqdm
import os
import random, sys
from src.city import City
from src.create_images import create_images
from UserInputs import UserInputs
from PIL import Image
from outputs.dataframe import DF


# delete all the photos, making room for analysis of new city
# DELETE ALL PHOTOS BEFORE PUSHING TO GIT !!
# DELETE GOOGLE API KEY AS WELL BEFORE PUSHING
def delete_photos(city = None):
    try:
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
        for file in os.listdir(UserInputs.ALTERED_IMG_PATH):
            os.remove(UserInputs.ALTERED_IMG_PATH + file)
        for file in os.listdir(UserInputs.ROOFS_IMG_PATH):
            os.remove(UserInputs.ROOFS_IMG_PATH + file)
        for file in os.listdir(UserInputs.FINAL_ROOFS_IMG_PATH):
            os.remove(UserInputs.FINAL_ROOFS_IMG_PATH + file)
        for file in os.listdir(UserInputs.TREES_IMG_PATH):
            os.remove(UserInputs.TREES_IMG_PATH + file)
        if city:
            for file in os.listdir(UserInputs.CITY_PATH + city + '/'):
                os.remove(UserInputs.CITY_PATH + city + '/' + file)

        print('Photos successfully deleted')

    except OSError as err:
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not delete photos.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


# main function
if __name__ == '__main__':

    all_columns = UserInputs.DEFAULT_COLUMNS + ['Albedo', 'Roofs (mi^2)', 'Greenery (%)', 'Trees (%)', 'Area Calculated (%)']

    data = DF(all_columns, UserInputs.DEFAULT_COLUMNS, UserInputs.DEFAULT_SCRAPING_URL)

    data.add_city_values('Stockton')

    # el paso coordinates: 31.7619, -106.4850
    # stockon coordinates: 37.9577, -121.2908
    stockton = City('Stockton', [37.9577, -121.2908], 5, data.df, data.return_row("Stockton"))


    for i in tqdm(range(3)):

        delete_photos(city="Stockton")
        # # stockton.find_raw_images(stockton.batch_size, new_images=False)
        # stockton.find_raw_images(stockton.batch_size)
        # # stockton.crop_images()
        # stockton.calculate_albedo()
        # stockton.find_greenery()
        # stockton.remove_color(UserInputs.LOW_GREEN, UserInputs.HIGH_GREEN)
        # stockton.remove_color(UserInputs.LOW_YELLOW, UserInputs.HIGH_YELLOW)
        # stockton.alter_images(otsu=False, sharpen=False)
        #
        # stockton.find_roofs()
        # stockton.calculate_roofs()
        # stockton.calculate_trees()
        # # stockton.find_contours()
        # stockton.percent_green()
        #
        # stockton.integrate(data.df)
        # data.print_df()

    # data.write_excel()


# results for El Paso should be: 21.583 miles of roofing, ~5% tree coverage, albedo in the low to mid 20s
