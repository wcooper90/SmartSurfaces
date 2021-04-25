from PIL import Image
import cv2
import numpy as np
import math
import os
import sys
sys.path.append(os.path.abspath('../'))
from UserInputs import UserInputs


def sharp(gray):
    blur = cv2.bilateralFilter(gray, 5, sigmaColor=7, sigmaSpace=5)
    kernel_sharp = np.array((
        [-2, -2, -2],
        [-2, 17, -2],
        [-2, -2, -2]), dtype='int')
    return cv2.filter2D(blur, -1, kernel_sharp)


def calculate_area(lat):
    metersPerPx = 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, int(UserInputs.DEFAULT_ZOOM))
    return metersPerPx
