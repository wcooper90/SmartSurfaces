import cv2
import numpy as np

# from stackoverflow
class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unidentified"

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "triangle"

        elif len(approx) == 4:
            print("value of approx", approx)
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            print("value of ar",ar)
            if (ar >= 0.95 and ar <= 1.05): shape = "Square"
            elif (ar <= 5 and ar >= 3): shape = "Obround"
            else: shape = "rectangle"

        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 2:
            shape = "line"
            print("value of approx", approx)
        else:
            shape = "circle"
            print("value of approx", approx)

        return shape
