# User input class, to be used across all files
import os


class UserInputs():
    """
    A class which allows the user to select type of input, which
    text-summarization techniques to use, and parameters to specify output
    form and type.
    """
    # image globals
    DEFAULT_HEIGHT = 640
    DEFAULT_WIDTH = 640
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
    ROOFS_IMG_PATH = PATH + 'images/analysis/roofs/raw/'
    FINAL_ROOFS_IMG_PATH = PATH + 'images/analysis/roofs/final/'
    TREES_IMG_PATH = PATH + 'images/analysis/trees/final/'
    TREES_RAW_PATH = PATH + 'images/analysis/trees/raw/'
    CITY_PATH = PATH + 'images/cities/'


    # default URL for website with data on US cities
    DEFAULT_SCRAPING_URL = 'https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population'

    # unit conversion, square miles and square meters to square feet
    SMILES_TO_SFEET = 27878000
    SMETERS_TO_SFEET = 10.7639

    # default columns for webscraping
    DEFAULT_COLUMNS = ['City', 'Population', 'Area (mi^2)', 'Location']

    # HSV values for color removal
    # LOW_YELLOW = (10, 10, 10)
    # HIGH_YELLOW = (45, 80, 170)
    # LOW_GREEN = (45, 20, 10)
    # HIGH_GREEN = (95, 255, 250)

    LOW_YELLOW = (10, 20, 30)
    HIGH_YELLOW = (60, 150, 220)
    LOW_GREEN = (24, 0, 0)
    HIGH_GREEN = (100, 250, 240)

    # LOW_YELLOW = (10, 10, 12)
    # HIGH_YELLOW = (60, 150, 220)
    # LOW_GREEN = (24, 0, 0)
    # HIGH_GREEN = (96, 240, 220)

    # margins of city coordinates, so can determine where to take images from
    CITY_MARGINS = 0.01
    # default zoom for the google maps downloader
    DEFAULT_ZOOM = str(20)
    # coordinates accuracy, round to how many decimals
    ZOOM_DECIMALS = 6

    # brightness, contrast, sharpness increase for roof finding
    BRIGHTNESS_INCREASE = 0.95
    CONTRAST_INCREASE = 2.5
    SHARPNESS_INCREASE = 1.1

    # random number for anything in the script that needs randomness
    RANDOM_SEED = 36558

    # threshold for biggest and smallest roofs allowed (in pixels)
    MAX_WIDTH_LENGTH = 450
    MIN_WIDTH_LENGTH = 30
    MAX_TREE_LENGTH = 450
    MIN_TREE_LENGTH = 10

    # Entropy threshold, increased for El Paso, decreased for Stockton
    MAX_ENTROPY = 2.7
    MIN_ENTROPY = 0.6
    # MAX_ENTROPY = 1.95
