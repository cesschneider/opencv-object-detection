# importing modules

import cv2
import numpy as np

# capturing video through webcam
cap = cv2.VideoCapture(1)

while (1):
    _, img = cap.read()

    # converting frame(img i.e BGR) to HSV (hue-saturation-value)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    '''
[14, 14, 113] RED
BLUE [87,13,0], [126,255,255]
ORANGE 
[104, 13, 140] GREEN
    
    orange
    [11, 161, 241]
    [11, 166, 255]
    [11, 208, 255]
    [12, 144, 184]    
    [12, 156, 234]
    [12, 162, 252]
    [12, 167, 255]
    [12, 168, 255]
    [13, 170, 255]
    [13, 178, 255]
    [14, 168, 255]
    [14, 172, 255]
    [14, 182, 255]
    [14, 180, 255]
    
    red
    [8, 110, 193]
    [8, 122, 111]
    [9, 103, 200]
    [9, 95, 196]
    [9, 108, 194]
    [9, 104, 172]
    [9, 129, 150]
    [9, 92, 202]
    [10, 95, 150]
    [10, 124, 161]
    [10, 118, 194]
    [10, 99, 203]
    [10, 110, 190]
    [11, 91, 182]
    [11, 100, 173]
    [11, 122, 196]
    [11, 140, 226]
    [11, 153, 255]
    [11, 155, 255
    '''

    red_lower = np.array([0, 50, 50], np.uint8)
    red_upper = np.array([6, 255, 150], np.uint8)

    orange_lower = np.array([10, 150, 200], np.uint8)
    orange_upper = np.array([16, 255, 255], np.uint8)

    #104, 13, 140
    green_lower = np.array([65, 100, 80], np.uint8)
    green_upper = np.array([95, 180, 150], np.uint8)

    blue_lower = np.array([99, 115, 150], np.uint8)
    blue_upper = np.array([110, 255, 255], np.uint8)

    yellow_lower = np.array([22, 60, 130], np.uint8)
    yellow_upper = np.array([60, 200, 200], np.uint8)

    white_lower = np.array([0, 0, 150], np.uint8)
    white_upper = np.array([10, 10, 255], np.uint8)

    orange = cv2.inRange(hsv, orange_lower, orange_upper)
    white = cv2.inRange(hsv, white_lower, white_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)
    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Morphological transformation, Dilation
    kernal = np.ones((5, 5), "uint8")

    area_min = 1000
    area_max = 10000

    red = cv2.dilate(red, kernal)
    res = cv2.bitwise_and(img, img, mask=red)
    cv2.imshow("RED", res)

    orange = cv2.dilate(orange, kernal)
    res_orange = cv2.bitwise_and(img, img, mask=orange)
    #   cv2.imshow("ORANGE", res_orange)

    white = cv2.dilate(white, kernal)
    res_white = cv2.bitwise_and(img, img, mask=white)
    cv2.imshow("WHITE", res_white)

    green = cv2.dilate(green, kernal)
    res_green = cv2.bitwise_and(img, img, mask=green)
    #cv2.imshow("green res", res_green)

    blue = cv2.dilate(blue, kernal)
    res1 = cv2.bitwise_and(img, img, mask=blue)
    cv2.imshow("BLUE", res1)

    yellow = cv2.dilate(yellow, kernal)
    res2 = cv2.bitwise_and(img, img, mask=yellow)
    #cv2.imshow("yellow res", res2)

    # Tracking the orange Color
    (contours, hierarchy) = cv2.findContours(orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #print('orange', area)
        if (area > area_min and area < area_max):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (85, 165, 255), 2)
            cv2.putText(img, "ORANGE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 165, 255))

    # Tracking the white Color
    (contours, hierarchy) = cv2.findContours(white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #print('white', area)
        if (area > area_min and area < area_max):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, "WHITE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    # Tracking the Green Color
    (contours, hierarchy) = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #print('green', area)
        if (area > area_min and area < area_max):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (141, 171, 20), 2)
            cv2.putText(img, "GREEN", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (141, 171, 20))

    # Tracking the Red Color
    (contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #print('red', area)
        if (area > area_min and area < area_max):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "RED", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Tracking the Blue Color
    (contours, hierarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #print('blue', area)
        if (area > area_min and area < area_max):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "BLUE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    # Tracking the yellow Color
    (contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > area_min and area < area_max):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "YELLOW", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # cv2.imshow("Redcolour",red)
    cv2.imshow("Color Tracking", img)
    # cv2.imshow("red",res)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


