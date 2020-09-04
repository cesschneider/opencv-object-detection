# importing modules

import cv2
import numpy as np

# capturing video through webcam
cap = cv2.VideoCapture(0)
square = None
filters = []

def on_mouse_click (event, x, y, flags, frame):
    global square
    if event == cv2.EVENT_LBUTTONUP:
        # print color values from HSV color space
        print(frame[y, x].tolist())
        square = [y, x]

        for iy in range(y - 2, y + 2):
            for ix in range(x - 2, x + 2):
                print(frame[iy, ix].tolist())
                filters.append(frame[iy, ix].tolist())


while (1):
    ws = 20
    _, img = cap.read()

    # converting frame(img i.e BGR) to HSV (hue-saturation-value)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)
    cv2.setMouseCallback('hsv', on_mouse_click, hsv)

    if square is not None:
        syl = square[0] - 10  # (ws / 2)
        syu = square[0] + 10  # (ws / 2)
        sxl = square[1] - 10  # (ws / 2)
        sxu = square[1] + 10  # (ws / 2)
        cv2.imshow('square', hsv[syl:syu, sxl:sxu])

    red_lower = np.array([0, 50, 50], np.uint8)
    red_upper = np.array([6, 255, 150], np.uint8)
    red = cv2.inRange(hsv, red_lower, red_upper)

    # Morphological transformation, Dilation
    kernel = np.ones((5, 5), "uint8")

    area_min = 1000
    area_max = 10000

    red = cv2.dilate(red, kernel)
    res = cv2.bitwise_and(img, img, mask=red)
    cv2.imshow("RED", res)

    # Tracking the Red Color
    (contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #print('red', area)
        if (area > area_min and area < area_max):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "RED", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    cv2.imshow("Color Tracking", img)
    key = cv2.waitKey(10)

    if key == ord('c'):
        filters = []

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


