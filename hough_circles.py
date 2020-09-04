import numpy as np
import cv2, os
from matplotlib import pyplot as plt

radius = [0, 255]

def detect_circles(path, filename, path_out):
    imgfile = path + '/' + filename

    ext = filename[-4:]
    # print(ext)
    if ext == 'jpg':
        name = filename[:-4]
    else:
        name = filename[25:-5]

    img = cv2.imread(imgfile, 0)
    cv2.imshow('original', img.copy())

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, radius[0], radius[1], cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresh', thresh)

    ret2, thresh2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh2', thresh2)

    ret3, thresh3 = cv2.threshold(img, 0, 255, cv2.cv2.THRESH_OTSU)
    cv2.imshow('thresh3', thresh3)

    edges = cv2.Canny(img, radius[0], radius[1])
    cv2.imshow('edges', edges)

    blur = cv2.medianBlur(img, 5)
    cv2.imshow('medianBlur', blur)

    #cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cimg = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    #circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=radius[0], maxRadius=radius[1])

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('output', cimg)

    key = cv2.waitKey(0)
    # print(key)
    factor = 1
    # ESC
    if key == 97:
        radius[0] = radius[0] - 1
    if key == 100:
        radius[0] = radius[0] + 1
    if key == 119:
        radius[1] = radius[1] - 1
    if key == 120:
        radius[1] = radius[1] + 1
    if key == 27:
        exit(0)
    # N or SPACE
    if key == 32 or key == 110:
        return True
    # S
    if key == 115:
        cv2.imwrite('{}/{}_LUV.jpg'.format(path_out, name), luv_full)

        # return True

    return False

date = '20200110'
#type = 'tissue red 2'  # 80-105 LUV (95)
type = 'tissue red 2'  # 15-25 B/R (20),
#type = 'empty 2'  # 25 R (remove edge), 95 LUV (ok)
# path = 'C:/Users/CesarSchneider/inveox GmbH/C2 - 20200102/white tissue'
# path_out = 'C:/Users/CesarSchneider/inveox GmbH/C2 - 20200102/empty_crop'
path = "E:/images/certa/c4.1/PoC/{}/{}".format(date, type)
path_out = "E:/images/certa/c4.1/PoC/{}/{} crop".format(date, type)
files = os.listdir(path)
i = 0

for file in files:
    i = i + 1
    if 0 < i < 2000:
        print(str(i) + ': ' + file)
        while detect_circles(path, file, path_out) is False:
            print(radius)
