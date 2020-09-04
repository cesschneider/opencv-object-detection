import cv2
import numpy as np
from time import sleep
from PIL import Image
from sampledetection import SampleDetection
from pypylon import pylon
from datetime import datetime

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, -0.5)

'''
[202, 228] [120] [70, 10, 70] has_sample? True 8572
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[202, 221] [124] [70, 10, 70] has_sample? False 88
[202, 221] [124] [70, 10, 70] has_sample? False 93
[202, 221] [124] [70, 10, 70] has_sample? False 25
[202, 221] [124] [70, 10, 70] has_sample? False 25
[202, 221] [124] [70, 10, 70] has_sample? False 0
[202, 221] [124] [70, 10, 70] has_sample? False 0
[202, 221] [124] [70, 10, 70] has_sample? True 216
[202, 218] [124] [70, 10, 70] has_sample? False 0


EMPTY
[202, 221] [124] [70, 10, 70] has_sample? False 0
[202, 221] [124] [70, 10, 70] has_sample? False 0
[202, 221] [124] [70, 10, 70] has_sample? False 25
[202, 221] [124] [70, 10, 70] has_sample? False 88
[202, 221] [124] [70, 10, 70] has_sample? False 93
[202, 221] [124] [70, 10, 70] has_sample? True 216

[202, 218] [124] [70, 10, 70] has_sample? False 0
[204, 218] [124] [70, 10, 71] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[205, 217] [127] [70, 10, 70] has_sample? False 0
[204, 218] [124] [70, 10, 71] has_sample? False 25
[204, 218] [124] [70, 10, 71] has_sample? False 30
[204, 218] [124] [70, 10, 71] has_sample? False 30
[204, 218] [124] [70, 10, 71] has_sample? False 60
[204, 218] [124] [70, 10, 71] has_sample? False 85
[204, 218] [124] [70, 10, 71] has_sample? False 101
[204, 218] [124] [70, 10, 71] has_sample? False 115
[204, 218] [124] [70, 10, 71] has_sample? False 133
[202, 218] [124] [70, 10, 70] has_sample? True 787 *
[202, 218] [124] [70, 10, 70] has_sample? True 814 *
[202, 218] [124] [70, 10, 70] has_sample? True 850 *

SMALL TISSUE (2x3mm)
[198, 218] [124] [70, 10, 70] has_sample? True 359
[198, 218] [124] [70, 10, 70] has_sample? True 375
[198, 218] [124] [70, 10, 70] has_sample? True 377
[198, 218] [124] [70, 10, 70] has_sample? True 359
[198, 218] [124] [70, 10, 70] has_sample? True 360
[204, 218] [124] [70, 10, 71] has_sample? True 396
[204, 218] [124] [70, 10, 71] has_sample? True 405
[202, 218] [124] [70, 10, 70] has_sample? True 417
[202, 218] [124] [70, 10, 70] has_sample? True 442
[204, 218] [124] [70, 10, 71] has_sample? True 428
[202, 218] [124] [70, 10, 70] has_sample? True 462
[202, 218] [124] [70, 10, 70] has_sample? True 462
[202, 218] [124] [70, 10, 70] has_sample? True 474
[202, 218] [124] [70, 10, 70] has_sample? True 441
[202, 218] [124] [70, 10, 70] has_sample? True 461
[202, 218] [124] [70, 10, 70] has_sample? True 451
[202, 218] [124] [70, 10, 70] has_sample? True 568

REALLY SMALL TISSUE
[202, 218] [124] [70, 10, 70] has_sample? False 30
[202, 218] [124] [70, 10, 70] has_sample? False 68
[202, 218] [124] [70, 10, 70] has_sample? False 66
[202, 218] [124] [70, 10, 70] has_sample? False 72
[202, 218] [124] [70, 10, 70] has_sample? False 86
[202, 218] [124] [70, 10, 70] has_sample? False 88
[202, 218] [124] [70, 10, 70] has_sample? False 140
[202, 218] [124] [70, 10, 70] has_sample? False 151
[202, 218] [124] [70, 10, 70] has_sample? False 161
[202, 218] [124] [70, 10, 70] has_sample? False 175
'''

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
    crop_img = img[320:700, 600:950]

    cv2.imshow('original', crop_img)

    sd = SampleDetection(crop_img, debug=True)
    sd.crop(coords[0], coords[1], radius[0])
    # sd.auto_crop(radius[0], debug=True)

    # ['FI', [rangeH[0], rangeS[0], rangeV[0]], [rangeH[1], rangeS[1], rangeV[1]], threshold[2]],

    # TODO: review DR1, DR2 color ranges (getting background noise)
    filters = [
        ['DR1', [39, 214, 0], [55, 255, 255], threshold[0]],
        ['DR2', [30, 210, 40], [55, 256, 100], threshold[0]],
        ['DR3', [0, 60, 5], [19, 255, 145], threshold[0]],
    ]
    debug = True
    for filter in filters:
        sd.filter(filter[0], [filter[1], filter[2]], threshold[0], debug)
        sd.transform(filter[0], filter[3], debug)

    # merge group of colours
    sd.merge('DR', [0, 1, 2], True)

    # TODO: review LR1, LR2 color ranges (too open, getting OR ranges)
    filters = [
        ['LR1', [0, 71, 0], [60, 255, 255], threshold[1]],
        ['LR2', [0, 50, 100], [39, 105, 230], threshold[1]],
    ]
    for filter in filters:
        sd.filter(filter[0], [filter[1], filter[2]], threshold[1], debug)
        sd.transform(filter[0], filter[3], debug)

    # merge group of colours
    sd.merge('LR', [3, 4], True)

    filters = [
        ['OR1', [0, 85, 185], [50, 190, 255], threshold[1]],
        ['OR2', [10, 100, 146], [25, 190, 255], threshold[1]],
        ['OR3', [10, 100, 146], [30, 255, 255], threshold[1]],
    ]
    for filter in filters:
        sd.filter(filter[0], [filter[1], filter[2]], threshold[1], debug)
        sd.transform(filter[0], filter[3], debug)

    # merge group of colours
    sd.merge('OR', [5, 6, 7], True)

    filters = [
        ['WT1', [0, 0, 180], [65, 70, 220], threshold[2]],
        ['WT2', [0, 0, 205], [33, 50, 255], threshold[2]],
        ['WT3', [15, 55, 172], [28, 90, 204], threshold[2]],
        ['WT4', [10, 35, 160], [28, 75, 218], threshold[2]],
        ['WT5', [140, 0, 245], [160, 10, 255], threshold[2]],
        ['WT6', [120, 10, 130], [140, 65, 195], threshold[2]],
        ['WT7', [135, 118, 119], [178, 142, 160], threshold[2]],
        ['WT8', [140, 124, 139], [165, 143, 159], threshold[2]],
    ]
    for filter in filters:
        #debug = False
        #if filter[0] == 'WT4':
        #    debug = True
        sd.filter(filter[0], [filter[1], filter[2]], threshold[2], debug)
        sd.transform(filter[0], filter[3], debug)

    # merge group of colours
    sd.merge('WT', [8, 9, 10, 11, 12, 13, 14, 15], True)

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
    '''

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
        print('{} {} {} has_sample? {} {}'.format(coords, radius, threshold, sd.has_sample(300), sd.density()))
        # cv2.imwrite('{}/{} {} {} {}.jpg'.format(path_out, timestamp, rangeH, rangeS, rangeV), sd.merged)

    # ESC
    if key == 27:
        exit(0)
