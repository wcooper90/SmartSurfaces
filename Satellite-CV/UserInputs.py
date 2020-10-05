# User input class, to be used across all files
import os


class UserInputs():
    """
    A class which allows the user to select type of input, which
    text-summarization techniques to use, and parameters to specify output
    form and type.
    """
    # globals
    NUM_IMAGES = 3
    DEFAULT_HEIGHT = 700
    DEFAULT_WIDTH = 700
    DEFAULT_IMAGE_TYPE = '.PNG'

    # standard folder path
    PATH = '/mnt/c/Users/wcoop/Desktop/SSC/Satellite-CV/'

    # image paths
    RAW_IMG_PATH = PATH + 'images/analysis/raw/'
    CROPPED_IMG_PATH = PATH + 'images/analysis/cropped/'
    GREEN_IMG_PATH = PATH + 'images/analysis/green/'
    ALTERED_IMG_PATH = PATH + 'images/analysis/altered/'
    CONTOURS_IMG_PATH = PATH + 'images/analysis/contours/'
    GRAY_IMG_PATH = PATH + 'images/analysis/gray/'

    # default URL for website with data on US cities
    DEFAULT_SCRAPING_URL = 'https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population'

    # unit conversion, square miles to square feet
    SMILES_TO_SFEET = 27878000

    # default columns for webscraping
    DEFAULT_COLUMNS = ['city', 'population', 'area', 'location']

    # HSV values for color removal
    LOW_YELLOW = (7, 0, 0)
    HIGH_YELLOW = (70, 140, 230)
    LOW_GREEN = (20, 10, 0)
    HIGH_GREEN = (100, 255,255)

    # margins of city coordinates, so can determine where to take images from
    CITY_MARGINS = 0.06

    # default zoom for the google maps downloader
    DEFAULT_ZOOM = 19

    # coordinates accuracy, round to how many decimals
    ZOOM_DECIMALS = 6

    # City coordinates
    COORDS = {"Boston": [42.3613, -71.0889, 42.3314, -71.0324]}
