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
    DEFAULT_HEIGHT = 500
    DEFAULT_WIDTH = 500
    DEFAULT_IMAGE_TYPE = '.PNG'

    # standard folder path
    PATH = '/mnt/c/Users/wcoop/Desktop/SSC/Satellite-CV/'

    # image paths
    RAW_IMG_PATH = PATH + 'images/raw/'
    CROPPED_IMG_PATH = PATH + 'images/cropped/'
    GREEN_IMG_PATH = PATH + 'images/green/'
    CONTOURS_IMG_PATH = PATH + 'images/contours/'
    GRAY_IMG_PATH = PATH + 'images/gray/'

    # default URL for website with data on US cities
    DEFAULT_SCRAPING_URL = 'https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population'

    # unit conversion, square miles to square feet
    SMILES_TO_SFEET = 27878000

    DEFAULT_COLUMNS = ['city', 'population', 'area', 'location']
