import cv2
import numpy as np
from time import sleep
from PIL import Image
from sampledetection import SampleDetection
from pypylon import pylon
from datetime import datetime

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

coords = [202, 228]
offset = [0, 0]
radius = [120]

#threshold = [20, 16, 120]
#threshold = [115, 16, 152]
threshold = [70, 10, 70]

rangeH = [10, 185]
rangeS = [0, 255]
rangeV = [160, 200]
# [10, 0, 160], [185, 255, 200]

date = '20200124'
path_in = 'E:/images/certa/c4.1/PoC/{}/input'.format(date)
path_out = 'E:/images/certa/c4.1/PoC/{}/output'.format(date)

while True:
    _, img = cap.read()
    crop_img = img[400:850, 200:650]

    cv2.imshow('original', crop_img)

    sd = SampleDetection(crop_img, debug=True)
    sd.crop(coords[0], coords[1], radius[0])
    # ['FI', [rangeH[0], rangeS[0], rangeV[0]], [rangeH[1], rangeS[1], rangeV[1]], threshold[2]],

    filters = [
        ['DR1', [39, 214, 0], [55, 255, 255], threshold[0]],
        ['DR2', [30, 210, 40], [55, 256, 100], threshold[0]],
        ['DR3', [0, 60, 5], [19, 255, 145], threshold[0]],
    ]
    for filter in filters:
        sd.filter(filter[0], [filter[1], filter[2]])
        sd.transform(filter[0], filter[3])

    # merge group of colours
    sd.merge('DR', [0, 1, 2])

    filters = [
        ['LR1', [0, 71, 0], [60, 255, 255], threshold[1]],
        ['LR2', [0, 50, 100], [39, 105, 230], threshold[1]],
        ['OR1', [0, 85, 185], [50, 190, 255], threshold[1]],
        ['OR2', [10, 100, 146], [25, 190, 255], threshold[1]],
        ['OR3', [10, 100, 146], [30, 255, 255], threshold[1]],
    ]
    for filter in filters:
        sd.filter(filter[0], [filter[1], filter[2]])
        sd.transform(filter[0], filter[3])

    # merge group of colours
    sd.merge('LR', [3, 4, 5, 6, 7])

    filters = [
        ['WT1', [0, 0, 180], [65, 70, 220], threshold[2]],
        ['WT2', [0, 0, 205], [33, 50, 255], threshold[2]],
        ['WT3', [15, 55, 172], [28, 90, 204], threshold[2]],
        ['WT4', [10, 35, 160], [28, 75, 218], threshold[2]]
    ]
    for filter in filters:
        sd.filter(filter[0], [filter[1], filter[2]])
        sd.transform(filter[0], filter[3])

    # merge group of colours
    sd.merge('WT', [8, 9, 10, 11], True)

    # put all filtered pixels into one image
    sd.merge_all()

    key = cv2.waitKey(1)
    # if key != -1:
    #    print(key)
    # +
    if key == 43:
        radius[0] = radius[0] + 1
    # -
    if key == 45:
        radius[0] = radius[0] - 1
    # I
    if key == 105:
        coords[1] = coords[1] + 1
    # M
    if key == 109:
        coords[1] = coords[1] - 1
    # J
    if key == 106:
        coords[0] = coords[0] + 1
    # K
    if key == 107:
        coords[0] = coords[0] - 1

    '''
    # A
    if key == 97:
        coords[0] = coords[0] + 1
    # D
    if key == 100:
        coords[0] = coords[0] - 1
    # W
    if key == 119:
        coords[1] = coords[1] + 1
    # X
    if key == 120:
        coords[1] = coords[1] - 1
    # F
    if key == 102:
        radius[0] = radius[0] - 1
    if key == 114:
        radius[0] = radius[0] + 1
    '''
    factor = 5
    # Q
    if key == 113:
        rangeH[0] = rangeH[0] - factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # W
    if key == 119:
        rangeH[0] = rangeH[0] + factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # A
    if key == 97:
        rangeS[0] = rangeS[0] - factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # S
    if key == 115:
        rangeS[0] = rangeS[0] + factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # Y
    if key == 121:
        rangeV[0] = rangeV[0] - factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # X
    if key == 120:
        rangeV[0] = rangeV[0] + factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # E
    if key == 101:
        rangeH[1] = rangeH[1] - factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # R
    if key == 114:
        rangeH[1] = rangeH[1] + factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # D
    if key == 100:
        rangeS[1] = rangeS[1] - factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # F
    if key == 102:
        rangeS[1] = rangeS[1] + factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # C
    if key == 99:
        rangeV[1] = rangeV[1] - factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))
    # V
    if key == 118:
        rangeV[1] = rangeV[1] + factor
        print('[{}, {}, {}], [{}, {}, {}]'.format(rangeH[0], rangeS[0], rangeV[0], rangeH[1], rangeS[1], rangeV[1]))

    # T
    if key == 116:
        threshold[0] = threshold[0] - 1
        print(threshold)
    # Z
    if key == 122:
        threshold[0] = threshold[0] + 1
        print(threshold)
    # G
    if key == 103:
        threshold[1] = threshold[1] - 1
        print(threshold)
    # H
    if key == 104:
        threshold[1] = threshold[1] + 1
        print(threshold)
    # B
    if key == 98:
        threshold[2] = threshold[2] - 1
        print(threshold)
    # N
    if key == 110:
        threshold[2] = threshold[2] + 1
        print(threshold)

    # O
    if key == 111:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} {} {} has_sample? {} {}'.format(coords, radius, threshold, sd.has_sample(300), sd.get_density()))
        # cv2.imwrite('{}/{} {} {} {}.jpg'.format(path_out, timestamp, rangeH, rangeS, rangeV), sd.merged)

    # ESC
    if key == 27:
        exit(0)
