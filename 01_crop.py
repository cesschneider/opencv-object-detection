import cv2
import numpy as np
import os
from PIL import Image
from sampledetection import SampleDetection
from time import sleep


def crop_image(path, filename, path_out):
    imgfile = path + '/' + filename

    ext = filename[-3:]
    # print(ext)
    if ext == 'jpg':
        name = filename[:-4]
    else:
        name = filename[25:-5]

    img = cv2.imread(imgfile, 0)
    img1 = cv2.imread(imgfile)
    img2 = cv2.imread(imgfile)

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Create mask
    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)

    edges = cv2.Canny(thresh, 100, 200)
    # cv2.imshow('detected ',gray)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1=50, param2=30, minRadius=0, maxRadius=0)
    # print(circles)
    # [[[208.5 206.5 180.8]]]
    # cv2.circle(img1, (200, 200), 160, (0, 255, 255), thickness=2)
    x, y, r = [coords[0] + offset[0], coords[1] + offset[1], radius[0]]
    #cv2.circle(img2, (x, y), r, (0, 255, 0), thickness=2)

    # Draw on mask
    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    #cv2.circle(mask, (int(i[0] + 5), int(i[1])), 90, (50, 50, 50), thickness=-1)

    # Copy that image using that mask
    masked_data = cv2.bitwise_and(img2, img2, mask=mask)

    # Apply Threshold
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Find Contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0][0])

    # Crop masked_data
    crop = masked_data[y:y + h, x:x + w]
    '''
    cv2.imshow('Detected circle',img2)
    '''
    cv2.imshow('Cropped', crop)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # print('{}/{} [{}, {}, {}].jpg'.format(path_out, name, coords[0], coords[1], radius[0]))
    cv2.imwrite('{}/{} [{}, {}, {}].jpg'.format(path_out, name, coords[0], coords[1], radius[0]), crop)
    # cv2.imwrite('{}/{}.jpg'.format(path_out, name), crop)


def detect_center(path, filename, path_out):
    gray = cv2.imread(path + '/' + filename, 0)
    #cv2.imshow('gray', gray)

    th, threshed = cv2.threshold(gray, 100, 255,
       cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # cv2.imshow('threshed', threshed)

    cnts = cv2.findContours(threshed, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    # print(cnts)

    # filter by area
    s1 = 10
    s2 = 100
    xcnts = []
    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            xcnts.append(cnt)

    # print(xcnts)
    img1 = cv2.imread(path + '/' + filename)

    cv2.circle(img1, (coords[0], coords[1]), 2, (0, 255, 255), thickness=3)
    cv2.circle(img1, (coords[0] + offset[0], coords[1] + offset[1]), radius[0], (0, 255, 255), thickness=2)
    '''
    for xcnt in xcnts:
        for i in xcnt:
            x, y = i[0]

            # def circle(img, center, radius, color, thickness=None, lineType=None,
            #           shift=None):  # real signature unknown; restored from __doc__

            cv2.circle(img1, (x, y), 1, (255, 0, 0), thickness=1)
            print(x, y)
    '''

    cv2.imshow('img1', img1)
    key = cv2.waitKey(0)
    # print('key:', key)

    if key == 97:
        coords[0] = coords[0] - 1
    if key == 100:
        coords[0] = coords[0] + 1
    if key == 119:
        coords[1] = coords[1] - 1
    if key == 120:
        coords[1] = coords[1] + 1
    if key == 101:
        radius[0] = radius[0] - 1
    if key == 114:
        radius[0] = radius[0] + 1
    if key == 32 or key == 110:
        return True
    if key == 115:
        crop_image(path, filename, path_out)
        # return True
    if key == 113 or key == 27:
        exit(0)

    return False


def crop_from_edges(path, filename, path_out):
    imgfile = path + '/' + filename

    ext = filename[-4:]
    # print(ext)
    if ext == 'jpeg':
        name = filename[:-4]
    else:
        name = filename[25:-5]

    img = cv2.imread(imgfile, 0)
    cv2.imshow('IN', img)

    # convert color to hsv because it is easy to track colors in this color model
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 0, 188])
    higher_hsv = np.array([179, 90, 255])

    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    # Apply the mask on the image to extract the original color
    frame_out = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('IN', img)
    cv2.imshow('OUT', frame_out)

    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=100, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
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
    # if key == 115:
        # return True

    return False


coords = [180, 211]
offset = [0, 0]
radius = [110]

threshold = [20, 16, 120]
#threshold = [115, 16, 152]
#threshold = [160, 16, 120]

date = '20200110'
#type = 'empty 2'  # 25 R (remove edge), 95 LUV (ok)
#type = 'tissue red 2'  # 80-105 LUV (95)
type = 'tissue white 2'  # 15-25 B/R (20),
#type = 'validation'  # 15-25 B/R (20),
# path = 'C:/Users/CesarSchneider/inveox GmbH/C2 - 20200102/white tissue'
path_out = 'C:\\Users\\CesarSchneider\\Desktop'
path = "E:/images/certa/c4.1/PoC/{}/{}".format(date, type)
path_out = "E:/images/certa/c4.1/PoC/{}/{} detected".format(date, type)
files = os.listdir(path)
i = 0

for file in files:
    i = i + 1
    if 17 < i < 2000:
        print(str(i) + ': ' + file)
        while sample_detection_debug(path, file, path_out) is False:
            print(coords, radius, threshold)

        #while crop_from_edges(path, file, path_out) is False:
        #    print(coords, radius)
        #while detect_center(path, file, path_out) is False:
        #    print(coords, radius)
        #while color_threshold(path, file, path_out) is False:
        #    print(threshold_param)

