import cv2, os
import numpy as np
from color_filter import color_filter_from_image, color_filter_from_list


def on_mouse_click (event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONUP:
        print(frame[y, x].tolist())


cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, -2.0)
#cap.set(cv2.CAP_PROP_CONTRAST, 0.5)


'''
white [0, 31, 33], [255, 0, 255]
white [0, 31, 15], [255, 132, 194]
W/R   [0, 47, 20], [255, 0, 255]
red   [0, 34, 43], [255, 0, 255]
red   [0, 47, 20], [255, 0, 105]
'''

filters_hsv = [
    [24, 55, 158],
    [24, 58, 162],
    [25, 75, 156],
    [21, 74, 156],
    [23, 69, 147],
    [27, 48, 137],
    [25, 62, 136],
    [23, 42, 151],
    [25, 64, 174],
    [16, 41, 160],
    [23, 50, 153],
    [23, 30, 137],
    [18, 40, 139],
    [11, 139, 55],
    [15, 127, 62],
    [17, 41, 162],
    [23, 58, 192],
    [26, 55, 187],
    [21, 57, 178],
    [15, 133, 46],
    [8, 101, 58],
    [24, 106, 82],
    [19, 108, 87],
    [20, 105, 90],
    [23, 173, 78]
]


while True:
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 0)
    _, img1 = cap.read()
    #img1 = img1[360:720, 146:520]
    cv2.imshow('crop_img 1', img1)

    cap.set(cv2.CAP_PROP_BRIGHTNESS, -10.0)
    cap.set(cv2.CAP_PROP_CONTRAST, 5.0)
    cap.set(cv2.CAP_PROP_SATURATION, 5.0)
    _, img2 = cap.read()
    #img2 = img2[360:720, 146:520]
    cv2.imshow('crop_img 2', img2)

    crop_img = img1.copy()
    cv2.setMouseCallback('crop_img', on_mouse_click, crop_img)

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV_FULL)
    cv2.imshow('hsv', hsv)
    cv2.setMouseCallback('hsv', on_mouse_click, hsv)

    color_filter_from_list(crop_img, filters_hsv)
    # color_filter_from_list(crop_img, filters_rgb)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


