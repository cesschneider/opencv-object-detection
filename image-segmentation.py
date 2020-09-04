import numpy as np
import cv2, os
from matplotlib import pyplot as plt


def image_segmentation(path, filename, path_out):
    imgfile = path + '/' + filename

    ext = filename[-4:]
    # print(ext)
    if ext == 'jpg':
        name = filename[:-4]
    else:
        name = filename[25:-5]

    img = cv2.imread(imgfile)
    cv2.imshow('original', img.copy())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresh', thresh)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow('opening', opening)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imshow('sure_bg', sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    cv2.imshow('sure_bg', sure_bg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    cv2.imshow('unknown', unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,0,255]
    #print(markers)
    cv2.imshow('output', img)

    key = cv2.waitKey(0)
    # print(key)
    factor = 1
    # ESC
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
        while image_segmentation(path, file, path_out) is False:
            print(str(i) + ': ' + file)

