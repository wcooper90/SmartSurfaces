from __future__ import print_function
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
from shapely.geometry import Polygon

zoom = 19
tileSize = 256
initialResolution = 2 * math.pi * 6378137 / tileSize
originShift = 2 * math.pi * 6378137 / 2.0
earthc = 6378137 * 2 * math.pi
factor = math.pow(2, zoom)
map_width = 256 * (2 ** zoom)


def grays(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
def white_image(im):
    return cv2.bitwise_not(np.zeros(im.shape, np.uint8))
def pixels_per_mm(lat):
    metersPerPx = np.abs(156543.03392 * math.cos(np.abs(lat) * math.pi / 180) / math.pow(2, zoom))
    return metersPerPx
def sharp(gray):
    blur = cv2.bilateralFilter(gray, 5, sigmaColor=7, sigmaSpace=5)
    kernel_sharp = np.array((
        [-2, -2, -2],
        [-2, 17, -2],
        [-2, -2, -2]), dtype='int')
    return cv2.filter2D(blur, -1, kernel_sharp)



def contours_canny(cnts):
    cv2.drawContours(canny_contours, cnts, -1, 255, 1)

    # Removing the contours detected inside the roof
    for cnt in cnts:
        counters = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []

        if cv2.contourArea(cnt) > 10:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counters += 1
                    pts.append((x, y))

        if counters > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(canny_polygons, [pts], True, 0)


def contours_img(cnts):
    cv2.drawContours(image_contours, cnts, -1, 255, 1)

    # Removing the contours detected inside the roof
    for cnt in cnts:
        counter = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []
        if cv2.contourArea(cnt) > 5:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counter += 1
                    pts.append((x, y))
        if counter > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(image_polygons, [pts], True, 0)



if __name__ == "__main__":
    images = glob.glob('testcases/*')
    # latitude = ??
    # pl, pw, l, w, solar_angle = solar_panel_params()
    # length, width = pixels_per_mm(latitude)

    for fname in images:
    	# pl = No of panels together as length commonside, pw = Same as for pw here w = width
    	# l = Length of panel in mm, w = Width of panel in mm
    	# solar_angle = Angle for rotation
        pl, pw, l, w, solar_angle = 4, 1, 8, 5, 30
        image = cv2.imread(fname)



        img = cv2.pyrDown(image)
        print('image shape : ',img.shape)
        n_white_pix = np.sum(img==255)
        print('num of white pixels : ',n_white_pix)
        # Upscaling of Image
        high_reso_orig = cv2.pyrUp(image)



        # White blank image for contours of Canny Edge Image
        canny_contours = white_image(image)
        # White blank image for contours of original image
        image_contours = white_image(image)



        # White blank images removing rooftop's obstruction
        image_polygons = grays(image_contours)
        canny_polygons = grays(canny_contours)

        im = Image.fromarray(canny_polygons)
        im.save("your_file.jpeg")

        # Gray Image
        grayscale = grays(image)

        sharp_image = sharp(grayscale)



        # Canny Edge
        edged = cv2.Canny(sharp_image, 180, 240)


        # Otsu Threshold (Automatic Threshold)
        thresh = cv2.threshold(sharp_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Contours in Original Image
        contours_img(cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2])
        # Contours in Canny Edge Image
        contours_canny(cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2])

        # Optimum place for placing Solar Panels
        solar_roof = cv2.bitwise_and(image_polygons, canny_polygons)
        plt.imshow(image_polygons, cmap='gray')
        plt.savefig('final.jpg')
        plt.imshow(canny_polygons, cmap='gray')
        plt.savefig('findal.jpg')
        plt.imshow(solar_roof, cmap='gray')
        plt.savefig('findalg.jpg')

        n_white_pix = np.sum(solar_roof==255)
        print(n_white_pix)
        area_roof = n_white_pix*pixels_per_mm(-121)
        print('area of building roof : ',area_roof,'sqm')
        print('size of solar roof : ',solar_roof.shape)
        new_image = white_image(image)

        plt.imshow(new_image, cmap='gray')
        plt.savefig('findsal.jpg')

        print('new image shape',new_image.shape)
        # Rotation of Solar Panels
        # panel_rotation(pl, solar_roof)

        plt.show()
