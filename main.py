import cv2
import pandas
import numpy as np
from tqdm import tqdm
import os
import random, sys
from src.objects.city import City
from src.create_images import create_images
from UserInputs import UserInputs
from PIL import Image
from src.objects.dataframe import DF


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
        for file in os.listdir(UserInputs.TREES_RAW_PATH):
            os.remove(UserInputs.TREES_RAW_PATH + file)
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

    # cities = {'Baltimore': ['detailed', 'geojson_path']}
    # non-detailed
    cities = {'Baltimore': [39.2382, -76.6037]}

                # 'Boston': [42.3601, -71.0589], 'Fresno': [36.7836,-119.7934]
                # 'El Paso': [31.7619, -106.4850], 'San Diego': [32.8153,-117.135],
                # 'Columbus': [39.9844,-82.9848], 'Memphis': [35.1028,-89.9774],
                # 'Washington': [38.9041, -77.0172],'Stockton': [37.9577, -121.2908]
                # 'Atlanta': [33.7629,-84.4227], 'Omaha': [41.2617,-96.0471],
                # 'Tampa': [27.9701,-82.4797], 'Washington': [38.9041, -77.0172],
                # 'Aurora': [39.688,-104.6897], 'Buffalo': [42.8925,-78.8597],
                # 'Richmond': [37.5314,-77.476], 'Montgomery': [32.3472,-86.2661],
                # 'Salt Lake City': [40.7769,-111.931], 'Providence': [41.8231,-71.4188],
                # 'Eugene': [44.0567,-123.1162], 'Joliet': [41.5177,-88.1488],
                # 'Charleston': [32.8179,-79.9589], 'Murfreesboro': [35.8481,-86.4088],
                # 'New Haven': [40.6885,-112.0118]}

    for key in cities:

        data.add_city_values(key)

        city = City(key, cities[key], 10, data.df, data.return_row(key))
        # delete_photos(city=key)

        for i in tqdm(range(1)):

            # city.find_raw_images(city.batch_size, new_images=False)
            city.find_raw_images(city.batch_size)
            # city.crop_images()
            city.calculate_albedo()
            city.standardize()
            city.find_greenery()
            city.remove_color(UserInputs.LOW_GREEN, UserInputs.HIGH_GREEN)
            city.remove_color(UserInputs.LOW_YELLOW, UserInputs.HIGH_YELLOW)
            city.alter_images(otsu=False, sharpen=False)
            city.find_roofs()
            city.calculate_roofs()
            # # city.find_trees()
            # city.calculate_trees()
            # # city.find_contours()
            city.percent_green()
            city.integrate(data.df)

            data.print_df()


    # data.write_excel()


# results for El Paso should be: 21.583 miles of roofing, ~5% tree coverage, albedo in the low to mid 20s
