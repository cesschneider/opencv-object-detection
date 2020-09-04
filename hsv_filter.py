import cv2, os
import numpy as np


cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
# h, s, v = 100, 100, 100


'''
-        'sampleAboveFilterHsvRangeTape': [(0, 0, 90), (255, 30, 115)],
+        'sampleAboveFilterHsvRangeTape': [(0, 0, 45), (255, 30, 65)],
'''

# Creating track bar
#cv2.createTrackbar('h', 'result', h, 179, nothing)
#cv2.createTrackbar('s', 'result', s, 255, nothing)
#cv2.createTrackbar('v', 'result', v, 255, nothing)

date = '20200110'
type = 'tissue red 2'
h, s, v = 0, 130, 0
#type = 'tissue white 1'
#h, s, v = 0, 0, 180
#type = 'tissue white 2'
#h, s, v = 0, 0, 130
#type = 'tissue white 3'
#h, s, v = 0, 0, 130

#path_out = "E:/images/certa/c4.1/current/{}/{}".format(date, type)
#path='C:/Users/CesarSchneider/inveox GmbH/C2 - 20191220/red white ring'
# path = "E:/images/certa/c4.1/PoC/{}/{}_crop".format(date, type)
path = "E:/images/certa/c4.1/PoC/{}/{} crop".format(date, type)
path_out = "E:/images/certa/c4.1/PoC/{}/{} filter".format(date, type)
files = os.listdir(path)
i = 0

for file in files:
    i = i + 1
    # convert_format(path, file, path_out)

    imgfile = path + '/' + file
    frame = cv2.imread(imgfile)

    ext = file[-3:]
    # print(ext)
    if ext == 'jpg':
        name = file[:-4]
    else:
        name = file[25:-5]
    # print(name)

    cv2.imshow('frame', frame)

    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    while 1:
        # get info from track bar and appy to result
        # h = cv2.getTrackbarPos('h','result')
        # s = cv2.getTrackbarPos('s','result')
        # v = cv2.getTrackbarPos('v','result')
        #cv2.setTrackbarPos('h', 'result', h)
        #cv2.setTrackbarPos('s', 'result', s)
        #cv2.setTrackbarPos('v', 'result', v)

        # Normal masking algorithm
        lower_blue = np.array([h,s,v])
        upper_blue = np.array([180,255,255])

        mask = cv2.inRange(hsv,lower_blue, upper_blue)
        result = cv2.bitwise_and(frame,frame,mask = mask)

        cv2.imshow('result', result)
        print(imgfile, h, s, v)

        key = cv2.waitKey(0)
        # print(key)
        if key == 27:
            exit(0)
        if key == 113:
            if h > 0:
                h = h - 1
        if key == 119:
            if h < 180:
                h = h + 1
        if key == 97:
            if s > 0:
                s = s - 1
        if key == 115:
            if s < 255:
                s = s + 1
        if key == 121:
            if v > 0:
                v = v - 1
        if key == 120:
            if v < 255:
                v = v + 1
        if key == 111:
            cv2.imwrite('{}/{}_HSV_{:03d}_{:03d}_{:03d}.jpg'.format(path_out, name, h, s, v), result)
        if key == 32 or key == 110:
            break
