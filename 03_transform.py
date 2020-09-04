import cv2
import numpy as np
import os
from PIL import Image
from sampledetection import SampleDetection


def sample_detection_debug(path, file, path_out):
    ext = file[-3:]
    # print(ext)
    if ext == 'jpg':
        name = file[:-4]
    else:
        name = file[25:-5]

    sd = SampleDetection(cv2.imread('{}/{}'.format(path, file)), debug=True)
    # sd.crop(coords[0], coords[1], radius[0])
    sd.auto_crop(radius[0], debug=True)

    '''
    ['LR1', [0, 71, 0], [60, 255, 255], threshold[0]],
    ['LR2', [0, 50, 100], [39, 105, 230], threshold[0]],
    ['DR1', [39, 214, 0], [55, 255, 255], threshold[1]],
    ['DR2', [30, 210, 40], [55, 256, 100], threshold[1]],
    ['DR3', [0, 60, 5], [19, 255, 145], threshold[1]],
    ['WT1', [0, 0, 0], [28, 255, 255], threshold[2]],
    ['WT2', [0, 0, 205], [33, 50, 255], threshold[2]],
    ['WT3', [15, 55, 172], [28, 90, 204], threshold[2]],
    ['WT4', [10, 35, 160], [28, 75, 218], threshold[2]],
    ['OR1', [0, 85, 185], [50, 190, 255], threshold[2]],
    ['OR2', [10, 100, 146], [25, 190, 255], threshold[2]],
    ['OR3', [10, 100, 146], [30, 255, 255], threshold[2]]
    '''
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
    sd.merge_all(threshold[2])

    key = cv2.waitKey(0)
    # print('key:', key)

    # C
    if key == 99:
        print(cv2.countNonZero(sd.output))

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
    # R
    if key == 114:
        radius[0] = radius[0] + 1

    # T
    if key == 116:
        threshold[0] = threshold[0] - 1
    # Z
    if key == 122:
        threshold[0] = threshold[0] + 1
    # G
    if key == 103:
        threshold[1] = threshold[1] - 1
    # H
    if key == 104:
        threshold[1] = threshold[1] + 1
    # B
    if key == 98:
        threshold[2] = threshold[2] - 1
    # N
    if key == 110:
        threshold[2] = threshold[2] + 1
    # SPACE
    if key == 32:
        return True

    # S
    if key == 115:
        cv2.imwrite('{}/{} ({}).jpg'.format(path_out, name, sd.density()), sd.cropped)
        cv2.imwrite('{}/{} [{}, {}, {}] [{}, {}, {}] ({}).jpg'.format(path_out, name,
                                                            coords[0], coords[1], radius[0],
                                                            threshold[0], threshold[1], threshold[2],
                                                            sd.density()), sd.output)
        # return True

    if key == 113 or key == 27:
        exit(0)

    return False

#threshold = [115, 16, 152]
#threshold = [160, 16, 120]

# date = '20200109'
# coords = [202, 206]
# radius = [122]
# type = 'tissue 2'
# threshold = [65, 80, 200]

# threshold = [65, 80, 12]
# [206, 200] [119] [65, 80, 12]
# 144: c4_1 (22980512)_20200109_182207250_0097.tiff

date = '20200110'
coords = [180, 211]
radius = [110]

# threshold = [100, 170, 200]
# type = 'empty 2'

# type = 'tissue red 2'
# threshold = [85, 152, 200]
# [180, 211] [110] [85, 152, 200] 9
# [180, 211] [110] [80, 171, 201] 11
# [180, 211] [110] [80, 141, 201] 108

type = 'tissue white 2'
threshold = [100, 152, 100]
# [180, 211] [110] [70, 140, 10]
# [180, 211] [110] [100, 152, 200]
# [180, 211] [110] [100, 210, 210] 55-59
# [180, 211] [110] [100, 210, 100] 60-70
# [195, 201] [125] [100, 152, 100]
# [192, 197] [127] [100, 152, 100]

path = "E:/images/certa/c4.1/PoC/{}/{}".format(date, type)
path_out = "E:/images/certa/c4.1/PoC/{}/{} transform".format(date, type)

files = os.listdir(path)
i = 0

for file in files:
    i = i + 1
    if 0 < i < 2000:
        print(str(i) + ': ' + file)
        while sample_detection_debug(path, file, path_out) is False:
            print(coords, radius, threshold)

