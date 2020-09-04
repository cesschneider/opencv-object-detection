import cv2, os
import numpy as np

'''
WHITE
1: 115553304_0001_159_171_110_HSV_000_000_140.jpg
[136, 155, 163]
[140, 145, 144]
[143, 149, 154]
[145, 152, 149]
[146, 148, 149]
[149, 161, 161]
[151, 160, 169]
[154, 174, 185]
[157, 161, 166]
[158, 164, 175]
[164, 166, 167]
[167, 177, 184]
[168, 187, 195]
1: 115947026_0001_156_170_108.jpg
[138, 139, 130]
[132, 129, 121]
[125, 129, 124]
[142, 147, 146]
[132, 135, 140]
[154, 171, 180]
[155, 172, 181]
[153, 167, 179]
[153, 165, 177]
[145, 167, 178]
[149, 168, 176]
207: 123621029_0102_160_172_107.jpg
[87, 140, 143]
[90, 155, 163]
[99, 149, 155]
[99, 161, 169]
[101, 145, 146]
[101, 148, 146]
[102, 160, 159]
43: 115553304_0043_154_171_110.jpg
[177, 194, 203]
[184, 202, 213]
[187, 211, 217]
[199, 216, 225]
[214, 228, 240]
[215, 224, 237]

RED
827: 125137231_0084_192_183_139_HSV_000_125_095.jpg
[33, 104, 77]
[36, 119, 97]
[40, 105, 83]
[51, 134, 109]
[51, 135, 110]
[59, 127, 102]
[75, 166, 143]
811: 125137231_0071_189_183_139_HSV_000_048_070.jpg
[3, 84, 51]
[6, 81, 53]
[4, 94, 64]
[16, 114, 84]
[19, 106, 78]
[52, 123, 96]
1: 123456398_0001_162_168_110.jpg
[15, 55, 54]
[16, 76, 76]
[16, 86, 85]
[18, 87, 76]
[24, 71, 62]
[26, 80, 67]
[26, 95, 85]
[27, 84, 86]
[43, 112, 109]
[51, 107, 96]
[61, 112, 98]
[76, 102, 86]
37: 123456398_0037_167_172_110.jpg
[17, 74, 76]
[18, 73, 82]
[19, 75, 92]
[19, 81, 87]
[24, 80, 81]
[32, 93, 97]
[32, 117, 127]
[36, 101, 116]
[40, 100, 106]
[40, 119, 132]
[61, 116, 119]
--
[3, 84, 51], [75, 166, 143]

--
NOT DETECTED
R 2 rectangle 625.0 0.575

'''


area_min = 100
area_max = 4000
areas = [10, 3000]
threshold = [150]
radius = [10, 800]
'''
20200102
RED 1   000-111 [0,120, 0], [40,255,255]
WHITE 1 000-104 [0, 0,134], [43, 43,255]
WHITE 1 105-216 [0,92,144], [43,255,255]
WHITE 2 000-221 [0, 0,131], [179,56,255]
WHITE 3 000-110 [0, 0,123], [179,61,255] container edge white
WHITE 3 000-110 [0, 0,158], [179,71,255] light white only (bottom)
WHITE 3 000-110 [0, 0,180], [179,71,255] light white only (right)

20200108
RED 1 0,40-60,0 60,255,255
WHITE 1 150,0,0 179,255,255 (right) whitish
WHITE 1 0,120-150,0 60,255,255 (left) redish

'''

rangeH = [0, 179]
rangeS = [0, 255]
rangeV = [0, 255]


def nothing(x):
    #print(x)
    print(cv2.getTrackbarPos('lowH', 'image'))
    pass


'''
    hls = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
    cv2.imshow('hls ', hls)

    hls_full = cv2.cvtColor(hsv_full, cv2.COLOR_BGR2HLS_FULL)
    # hls_full = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS_FULL)
    # gray_hls_full = cv2.cvtColor(hls_full, cv2.COLOR_BGR2GRAY)
    # ret3, thresh3 = cv2.threshold(gray_hls_full, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('HLS_FULL ', hls_full)
    # cv2.imshow('HLS_FULL gray', gray_hls_full)
    # cv2.imshow('HLS_FULL threshold', thresh3)

    luv = cv2.cvtColor(hsv_full, cv2.COLOR_BGR2LUV)
    # luv2 = cv2.cvtColor(img1, cv2.COLOR_BGR2LUV)
    gray_luv = cv2.cvtColor(luv, cv2.COLOR_BGR2GRAY)
    ret1, thresh1 = cv2.threshold(gray_luv, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('LUV ', luv)
    # cv2.imshow('LUV 2', luv2)
    cv2.imshow('LUV gray', gray_luv)
    cv2.imshow('LUV threshold 1', thresh1)

    lab1 = cv2.cvtColor(hsv_full, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    cv2.imshow('LAB 1', lab1)
    cv2.imshow('LAB 2', lab2)

    yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    cv2.imshow('YUV ', yuv)

    ycr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    cv2.imshow('YCR_CB ', ycr)

'''


def color_threshold(path, filename, path_out):
    original = cv2.imread(path + '/' + filename)
    ext = file[-3:]
    # print(ext)
    if ext == 'jpg':
        name = file[:-4]
    else:
        name = file[25:-5]

    # print(name)
    cv2.imshow('original', original)

    # PixelFormat	YCbCr422_8
    rgb = cv2.cvtColor(original, cv2.COLOR_YCR_CB2RGB)
    bgr = cv2.cvtColor(original, cv2.COLOR_YCrCb2BGR)
    cv2.imshow('original RGB', rgb)
    cv2.imshow('original BRG', bgr)
    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray rgb', gray_rgb)
    cv2.imshow('gray bgr', gray_bgr)

    b = original.copy()
    b[:, :, 1] = 0
    b[:, :, 2] = 0
    cv2.imshow('Blue ', b)

    br = original.copy()
    br[:, :, 1] = 0
    cv2.imshow('Blue Red', br)

    g = original.copy()
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    cv2.imshow('Green ', g)

    gb = original.copy()
    gb[:, :, 2] = 0
    cv2.imshow('Green Blue', gb)

    r = original.copy()
    r[:, :, 0] = 0
    r[:, :, 1] = 0
    cv2.imshow('Red ', r)

    r_rgb = rgb.copy()
    r_rgb[:, :, 0] = 0
    r_rgb[:, :, 1] = 0
    cv2.imshow('Red RGB', r_rgb)

    rg = original.copy()
    rg[:, :, 0] = 0
    cv2.imshow('Red Green', rg)

    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray_b', gray_b)
    ret_b, thresh_b = cv2.threshold(gray_b, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('thresh_b', thresh_b)

    gray_br = cv2.cvtColor(br, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray_br', gray_br)
    ret_br, thresh_br = cv2.threshold(gray_br, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('thresh_br', thresh_br)

    gray_g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray_g', gray_g)
    ret_g, thresh_g = cv2.threshold(gray_g, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('thresh_g', thresh_g)

    gray_r1 = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray_r1', gray_r1)
    gray_r2 = cv2.cvtColor(r_rgb, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray_r2', gray_r2)
    ret_r1, thresh_r1 = cv2.threshold(gray_r1, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('thresh_r1', thresh_r1)
    ret_r2, thresh_r2 = cv2.threshold(gray_r2, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('thresh_r2', thresh_r2)

    luv_full = cv2.cvtColor(original, cv2.COLOR_RGB2LUV)
    #luv_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2LUV)
    gray_luv_full = cv2.cvtColor(luv_full, cv2.COLOR_BGR2GRAY)
    ret_luv_full, thresh_luv_full = cv2.threshold(gray_luv_full, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('LUV', luv_full)
    #cv2.imshow('LUV gray', gray_luv_full)
    cv2.imshow('LUV threshold', thresh_luv_full)

    hsv_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)
    #hsv_bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV_FULL)
    gray_hsv_full = cv2.cvtColor(hsv_rgb, cv2.COLOR_BGR2GRAY)
    ret_hsv_full, thresh_hsv_full = cv2.threshold(gray_hsv_full, threshold_param[0], threshold_param[1], cv2.THRESH_BINARY)
    cv2.imshow('HSV RGB', hsv_rgb)
    #cv2.imshow('HSV BGR', hsv_bgr)
    #cv2.imshow('HSV_FULL gray', gray_hsv_full)
    cv2.imshow('HSV threshold', thresh_hsv_full)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((threshold_param[0], threshold_param[0]), np.uint8)
    img_copy = br.copy()

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(img_copy, kernel, iterations=1)
    img_dilation = cv2.dilate(img_copy, kernel, iterations=1)
    opening = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img_copy, cv2.MORPH_GRADIENT, kernel)
    ret_closing, thresh_closing = cv2.threshold(closing, threshold_param[1], 255, cv2.THRESH_BINARY)
    ret_gradient, thresh_gradient = cv2.threshold(gradient, threshold_param[1], 255, cv2.THRESH_BINARY)

    '''
    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    cv2.imshow('gradient', gradient)
    cv2.imshow('thresh_closing', thresh_closing)
    cv2.imshow('thresh_gradient', thresh_gradient)
    '''

    key = cv2.waitKey(0)
    # print(key)
    factor = 1
    # ESC
    if key == 27:
        exit(0)
    if key == 97:
        if threshold_param[0] > 0:
            threshold_param[0] = threshold_param[0] - factor
    if key == 100:
        if threshold_param[0] < 255:
            threshold_param[0] = threshold_param[0] + factor
    if key == 119:
        if threshold_param[1] < 255:
            threshold_param[1] = threshold_param[1] + factor
    if key == 120:
        if threshold_param[1] > 0:
            threshold_param[1] = threshold_param[1] - factor
    # N or SPACE
    if key == 32 or key == 110:
        return True
    # S
    if key == 115:
        cv2.imwrite('{}/{}_LUV.jpg'.format(path_out, name), luv_full)
        cv2.imwrite('{}/{}_HSV.jpg'.format(path_out, name), hsv_full)
        cv2.imwrite('{}/{}_R_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_r)
        cv2.imwrite('{}/{}_G_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_b)
        cv2.imwrite('{}/{}_B_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_r)
        cv2.imwrite('{}/{}_R_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_r)
        cv2.imwrite('{}/{}_G_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_b)
        cv2.imwrite('{}/{}_B_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_r)
        cv2.imwrite('{}/{}_LUV_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_luv_full)
        cv2.imwrite('{}/{}_HSV_{:03d}_{:03d}.jpg'.format(path_out, name, threshold_param[0], threshold_param[1]), thresh_hsv_full)
        # return True

    return False


def detect_color_shape(color_name, color_range, img_copy, debug=False):
    lower = np.array(color_range[0], np.uint8)
    upper = np.array(color_range[1], np.uint8)

    color = cv2.inRange(img_copy, lower, upper)
    kernel = np.ones((5, 5), 'uint8')

    color = cv2.dilate(color, kernel)
    #cv2.imshow('{} dilate'.format(color_name), color)
    masked = cv2.bitwise_and(img_copy, img_copy, mask=color)
    #cv2.imshow('{} masked'.format(color_name), masked)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('blurred', blurred)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN,kernel, iterations=2)
    cv2.imshow('opening', opening)

    #thresh = cv2.threshold(blurred, threshold[0], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(opening, threshold[0], 255, cv2.THRESH_BINARY)[1]

    # Find Contour
    (contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        #print(contour)
        cv2.drawContours(thresh, [contour], -1, (255, 0, 0), 2)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=radius[0], maxRadius=radius[1])
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(thresh, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(thresh, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('thresh', thresh)
    height, width, channels = img_copy.shape
    detected_image = np.zeros((height, width, 3), np.uint8)

    (contours, hierarchy) = cv2.findContours(color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        cv2.drawContours(img_copy, [contour], -1, (255, 0, 0), 2)
        area = cv2.contourArea(contour)

        #if debug is True:
        #    print('{} {} {}'.format(color_name, pic, area), end=' ')

        if (area > areas[0] and area < areas[1]):
            coords = []
            for c in contour:
                coords.append([c[0][0], c[0][1]])
            # print('contour {} {} {}'.format(pic, area, coords))
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(detected_image, [contour], -1, (0, 255, 0), cv2.FILLED)

            # TODO: calculate density of color inside bounding/contour
            ar = w / float(h)
            shape = "square" if ar >= 0.7 and ar <= 1.0 else "rectangle"
            #if debug is True:
            #    print('{} {}'.format(ar, shape))

            if color_name == 'W':
                color_bounding = (255, 255, 255)
            if color_name == 'R':
                color_bounding = (0, 0, 255)

            color_bounding = (255, 255, 255)

            img_copy = cv2.rectangle(img_copy, (x, y), (x + w, y + h), color_bounding, 2)
            #cv2.putText(img, "{} {}".format(color_name, pic), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bounding)
        else:
            if debug is True:
                print('')

    if debug is True:
        cv2.imshow('{} img_copy'.format(color_name), img_copy)
    #    cv2.imshow('{} DC'.format(color_name), detected_image)

    return detected_image


def color_filter_viewer(path, filename, path_out):
    frame = cv2.imread(path + '/' + filename)
    ext = filename[-3:]
    # print(ext)

    if ext == 'jpg':
        name = filename[:-4]
    else:
        name = filename[25:-5]

    #color_filter_from_image(img)

    '''
    # get current positions of the trackbars
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')
    '''

    # convert color to hsv because it is easy to track colors in this color model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([rangeH[0], rangeS[0], rangeV[0]])
    higher_hsv = np.array([rangeH[1], rangeS[1], rangeV[1]])

    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    # Apply the mask on the image to extract the original color
    frame_out = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('IN', frame)
    cv2.imshow('OUT', frame_out)

    key = cv2.waitKey(0)
    # print(key)
    factor = 1
    # ESC
    if key == 27:
        exit(0)
    # N or SPACE
    if key == 32 or key == 110:
        return True
    # O
    if key == 111:
        print('{}/{} {} {}.jpg'.format(path_out, name, lower_hsv, higher_hsv))
        cv2.imwrite('{}/{} {} {}.jpg'.format(path_out, name, lower_hsv, higher_hsv), frame_out)
    # Q
    if key == 113:
        rangeH[0] = rangeH[0] - factor
    # W
    if key == 119:
        rangeH[0] = rangeH[0] + factor
    # A
    if key == 97:
        rangeS[0] = rangeS[0] - factor
    # S
    if key == 115:
        rangeS[0] = rangeS[0] + factor
    # Y
    if key == 121:
        rangeV[0] = rangeV[0] - factor
    # X
    if key == 120:
        rangeV[0] = rangeV[0] + factor
    # E
    if key == 101:
        rangeH[1] = rangeH[1] - factor
    # R
    if key == 114:
        rangeH[1] = rangeH[1] + factor
    # D
    if key == 100:
        rangeS[1] = rangeS[1] - factor
    # F
    if key == 102:
        rangeS[1] = rangeS[1] + factor
    # C
    if key == 99:
        rangeV[1] = rangeV[1] - factor
    # V
    if key == 118:
        rangeV[1] = rangeV[1] + factor

    return False


def color_filter_from_image(img):

    # print(name)
    cv2.imshow('img', img)

    height, width, channels = img.shape
    blended_image = np.zeros((height, width, 3), np.uint8)

    img1 = detect_color_shape('W1', [[97, 165, 188], [108, 172, 190]], img.copy(), True)
    img2 = detect_color_shape('W2', [[125, 129, 124], [138, 139, 130]], img.copy(), True)  # container edges
    img3 = detect_color_shape('W3', [[87, 140, 143], [102, 160, 159]], img.copy(), True)
    #img4 = detect_color_shape('W4', [[136, 155, 163], [158, 187, 195]], img.copy())
    img4 = detect_color_shape('W4', [[146, 155, 163], [158, 187, 195]], img.copy(), True)  # some container edges
    img5 = detect_color_shape('W5', [[177, 194, 203], [215, 224, 237]], img.copy(), True)

    cv2.add(img1, blended_image, blended_image)
    cv2.add(img2, blended_image, blended_image)
    cv2.add(img3, blended_image, blended_image)
    cv2.add(img4, blended_image, blended_image)
    cv2.add(img5, blended_image, blended_image)
    #blended_image = cv2.addWeighted(img1, 1, blended_image, 0.8, 0)
    #blended_image = cv2.addWeighted(img4, 1, blended_image, 0.8, 0)
    #blended_image = cv2.addWeighted(img5, 1, blended_image, 0.8, 0)

    # detect_color_shape('R', [[3, 84, 51], [75, 166, 143]], img)
    img6 = detect_color_shape('R1', [[15, 55, 54], [27, 84, 86]], img.copy(), True)
    img7 = detect_color_shape('R2', [[24, 80, 81], [61, 116, 119]], img.copy(), True)  # some background and container edges

    cv2.add(img6, blended_image, blended_image)
    cv2.add(img7, blended_image, blended_image)
    cv2.imshow("blended_image", blended_image)

    key = cv2.waitKey(0)
    print(key)
    factor = 2
    # ESC
    if key == 27:
        exit(0)
    # N or SPACE
    if key == 32 or key == 110:
        return True
    # O
    if key == 111:
        cv2.imwrite('{}/{} {} {}.jpg'.format(path_out, name, lower_hsv, higher_hsv), frame_out)
    # Q
    if key == 113:
        if areas[0] > area_min:
            areas[0] = areas[0] - factor
    # W
    if key == 119:
        if areas[0] < area_max:
            areas[0] = areas[0] + factor
    # A
    if key == 97:
        if areas[1] > area_min:
            areas[1] = areas[1] - factor
    # S
    if key == 115:
        if areas[1] < area_max:
            areas[1] = areas[1] + factor
    if key == 121:
        threshold[0] = threshold[0] - factor
    if key == 120:
        threshold[0] = threshold[0] + factor

    return False


def color_filter_from_list(img, filters):

    # print(name)
    #cv2.imshow('img', img)

    height, width, channels = img.shape
    blended_image = np.zeros((height, width, 3), np.uint8)

    i = 0
    for filter in filters:
        filter_lower = [filter[0]-5, filter[1]-5, filter[2]-5]
        filter_upper = [filter[0]+5, filter[1]+5, filter[2]+5]
        img_result = detect_color_shape(i, [filter_lower, filter_upper], img.copy())
        cv2.add(img_result, blended_image, blended_image)
        i = i + 1

    cv2.imshow("blended_image", blended_image)


def on_mouse_click (event, x, y, flags, frame):
    print(event, x, y, flags, frame)
    if event == cv2.EVENT_LBUTTONUP:
        print(frame[y, x].tolist())
        # colors.append(frame[y, x].tolist())


if __name__ == '__main__':

    # Create a window
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.setMouseCallback('image', on_mouse_click, 1)
    '''
    # create trackbars for color change
    cv2.createTrackbar('lowH', 'image', 0, 179, nothing)
    cv2.createTrackbar('highH', 'image', 179, 179, nothing)

    cv2.createTrackbar('lowS', 'image', 0, 255, nothing)
    cv2.createTrackbar('highS', 'image', 255, 255, nothing)

    cv2.createTrackbar('lowV', 'image', 0, 255, nothing)
    cv2.createTrackbar('highV', 'image', 255, 255, nothing)

    cv2.setMouseCallback('lowH', on_mouse_click, 'lowH')
    cv2.setMouseCallback('highH', on_mouse_click, 'highH')

    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')
    '''

    '''
    20200108
    RED 1     0,  40-60, 0    60, 255, 255
    WHITE 1 150,      0, 0   179, 255, 255 (right) whitish
    WHITE 1   0,120-150,  0    60, 255, 255 (left) redish
    
    20200110
    red 2       0,160,255   60,255,255 (dark red, orange)
                
    white 2
    '''

    date = '20200109'
    type = 'tissue 2'
    # type = 'tissue red 2'
    # type = 'empty 2'

    # path = 'C:/Users/CesarSchneider/inveox GmbH/C2 - 20200102/white tissue'
    # path_out = 'C:/Users/CesarSchneider/inveox GmbH/C2 - 20200102/empty_crop'
    # path = "E:/images/lab/c2.3/{}/{}".format(date, type)
    # path_out = "E:/images/lab/c2.3/{}/{} filter".format(date, type)
    path = "E:/images/certa/c4.1/PoC/{}/{}".format(date, type)
    path_out = "E:/images/certa/c4.1/PoC/{}/{} filter".format(date, type)

    files = os.listdir(path)
    i = 0

    for file in files:
        i = i + 1
        if 17 < i < 2000:
            print(str(i) + ': ' + file)
            img = cv2.imread(path + '/' + file)

            #while color_filter_from_image(img) is False:
            while color_filter_viewer(path, file, path_out) is False:
                #print(areas[0], areas[1])
                #print(threshold)
                print("[{}, {}, {}], [{}, {}, {}]".format(rangeH[0], rangeS[0], rangeV[0],
                                                          rangeH[1], rangeS[1], rangeV[1]))


