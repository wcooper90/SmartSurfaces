# User input class, to be used across all files
import os


class UserInputs():
    """
    A class which allows the user to select type of input, which
    text-summarization techniques to use, and parameters to specify output
    form and type.
    """
    NUM_IMAGES = 3
    RAW_IMG_PATH = '/mnt/c/Users/wcoop/Desktop/SSC/Satellite-CV/images/raw/'
    CROPPED_IMG_PATH = '/mnt/c/Users/wcoop/Desktop/SSC/Satellite-CV/images/cropped/'
    GREEN_IMG_PATH = '/mnt/c/Users/wcoop/Desktop/SSC/Satellite-CV/images/green/'
    CONTOURS_IMG_PATH = '/mnt/c/Users/wcoop/Desktop/SSC/Satellite-CV/images/contours/'
    GRAY_IMG_PATH = '/mnt/c/Users/wcoop/Desktop/SSC/Satellite-CV/images/gray/'
    DEFAULT_HEIGHT = 500
    DEFAULT_WIDTH = 500
    DEFAULT_IMAGE_TYPE = '.PNG'
