import cv2, os, numpy as np


class SampleDetection:

    def __init__(self, image, debug=False):
        self.debug = debug
        self.original = image
        # self.cropped = None
        self.cropped = image
        self.output = None
        self.groups = []
        self.filtered = []
        self.transformed = []
        self.merged = []
        self.data = []

    def crop(self, x, y, r):
        # Create mask
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        mask = np.zeros((height, width), np.uint8)

        # Draw on mask
        cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)

        # Copy that image using that mask
        masked_data = cv2.bitwise_and(self.original, self.original, mask=mask)

        # Apply Threshold
        threshold = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Find Contour
        contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0][0])

        # Crop masked_data
        self.cropped = masked_data[y:y + h, x:x + w]

        if self.debug:
            cv2.imshow('Cropped', self.cropped)

        return True

    def auto_crop(self, radius, debug=False):
        image = self.original

        firstCropSize = 200
        firstCropCenter = (200, 200)
        radius = 270
        thickness = 140

        # crop the image and draw a circle to remove outside background
        image = image[firstCropCenter[0] - firstCropSize:firstCropCenter[0] + firstCropSize,
                firstCropCenter[1] - firstCropSize:firstCropCenter[1] + firstCropSize, :]
        cv2.circle(image, firstCropCenter, radius, (0, 0, 0), thickness=thickness)

        if debug:
            cv2.imshow('first crop', image)
        enhanced = self.enhaneContrast(image)

        # enhanced = image
        if debug:
            cv2.imshow('enhanced', enhanced)

        binary = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        threshold = cv2.threshold(binary, 180, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((8, 8))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        if debug:
            cv2.imshow('thresh', threshold)

        cnts, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        maxCnt = np.reshape(cnts[max_index], (len(cnts[max_index]), 2))
        firstPoint = sorted(maxCnt, key=lambda k: k[1])[0]
        offset = 225
        points = []

        for point in maxCnt:
            if point[1] == firstPoint[1] + offset:
                points.append(point)

        points.sort(key=lambda k: k[0])
        points = [points[0], points[-1]]
        centerPoint, radius2 = self.define_circle(firstPoint, points[0], points[1])

        '''
        points = [points[0], points[-1]]
        IndexError: list index out of range
        '''

        if debug:
            print('points', points)
            cv2.circle(image, (firstPoint[0], firstPoint[1]), 5, (20, 50, 120), thickness=5)
            cv2.circle(image, (int(centerPoint[0]), int(centerPoint[1])), 5, (200, 20, 120), thickness=5)
            cv2.circle(image, (points[0][0], points[0][1]), 5, (120, 150, 20), thickness=5)
            cv2.circle(image, (points[1][0], points[1][1]), 5, (120, 150, 20), thickness=5)
            print('circle', centerPoint, radius2)

        centerPoint = (int(centerPoint[0]), int(centerPoint[1]))
        thickness2 = 80
        cropReduction = 20
        # size = 180
        size = radius

        # cv2.circle(image, centerPoint, int(radius2), (20, 50, 120), thickness=2)
        cv2.circle(image, centerPoint, int(radius2 + thickness2 / 2) - cropReduction, (0, 0, 0), thickness=thickness2)
        finalCrop = image[centerPoint[1] - size:centerPoint[1] + size, centerPoint[0] - size:centerPoint[0] + size]

        if debug:
            cv2.imshow('final', finalCrop)

        self.cropped = finalCrop

    def enhaneContrast(self, img):
        # -----Converting image to LAB Color model-----------------------------------
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))
        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def define_circle(self, p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
        if abs(det) < 1.0e-6:
            return (None, np.inf)
        # Center of circle
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
        return ((cx, cy), radius)

    def filter(self, color, range, level=100, debug=False):
        # Convert color to hsv because it is easy to track colors in this color model
        hsv = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array(range[0])
        higher_hsv = np.array(range[1])

        # Apply the cv2.inrange method to create a mask
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

        # Apply the mask on the image to extract the original color
        filtered = cv2.bitwise_and(self.cropped, self.cropped, mask=mask)
        threshold = cv2.threshold(filtered, level, 255, cv2.THRESH_BINARY)[1]
        # self.filtered.append(filtered)
        self.filtered.append(threshold)

        if debug:
            cv2.imshow('FIC {}'.format(color), filtered)
            # cv2.imshow('FIT {}'.format(color), threshold)

    def transform(self, color, level=100, debug=False):
        index = len(self.filtered) - 1
        gray = cv2.cvtColor(self.filtered[index], cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3, 3), np.uint8)
        # erosion = cv2.erode(blurred, kernel, iterations=1)
        dilation = cv2.dilate(gray, kernel, iterations=1)
        # if debug:
        #    cv2.imshow('DI {}'.format(color), dilation)

        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # GaussianBlur = cv2.GaussianBlur(dilation, (3, 3), 0)
        # if debug:
        #    cv2.imshow('GB {}'.format(color), GaussianBlur)

        # bilateralFilter = cv2.bilateralFilter(dilation, 9, 75, 75)
        # if debug:
        #    cv2.imshow('BF {}'.format(color), bilateralFilter)

        medianBlur = cv2.medianBlur(dilation, 5)
        if debug:
            cv2.imshow('MB {}'.format(color), medianBlur)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(medianBlur, cv2.MORPH_OPEN, kernel, iterations=2)

        # close holes
        closing = cv2.morphologyEx(medianBlur, cv2.MORPH_CLOSE, kernel, iterations=5)

        #if debug:
        #    cv2.imshow('MO {}'.format(color), opening)
        #    cv2.imshow('MC {}'.format(color), closing)

        threshold = cv2.threshold(closing, level, 255, cv2.THRESH_BINARY)[1]
        self.transformed.append(threshold)
        # self.transformed.append(closing)

    def merge(self, group, indexes, debug=False):
        #index = len(self.filtered) - 1
        #merge_index = len(self.transformed) - 1
        gray = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2GRAY)
        #gray = cv2.cvtColor(self.filtered[merge_index], cv2.COLOR_BGR2GRAY)
        #gray = self.transformed[merge_index]
        height, width = gray.shape
        merged = np.zeros((height, width), np.uint8)

        self.groups.append(group)
        group_index = len(self.groups) - 1

        for image_index in indexes:
            #cv2.add(cv2.cvtColor(self.filtered[index], cv2.COLOR_BGR2GRAY), merged, merged)
            cv2.add(self.transformed[image_index], merged, merged)

        self.merged.append(merged)
        # blurred = cv2.GaussianBlur(merged, (5, 5), 0)
        # blurred = cv2.medianBlur(dilation, 5)
        # self.merged.append(blurred)

        if debug:
            cv2.imshow('MG ' + group, merged)

    def merge_all(self, level=100, identify=True):
        gray = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        self.output = np.zeros((height, width), np.uint8)

        g = 0
        for image in self.merged:
            binary = cv2.threshold(image, level, 255, cv2.THRESH_BINARY)[1]
            cv2.add(binary, self.output, self.output)

        '''
            (contours, hierarchy) = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for index, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                print(area, end=' ')

                if self.groups[g] == 'DR':
                    color = (0, 0, 150)
                if self.groups[g] == 'LR':
                    color = (0, 0, 255)
                if self.groups[g] == 'OR':
                    color = (0, 150, 150)
                if self.groups[g] == 'WT':
                    color = (255, 255, 255)

                if 100 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    img = cv2.rectangle(self.cropped, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(self.cropped, self.groups[g], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            g = g + 1

        print('')
        '''

        (contours, hierarchy) = cv2.findContours(self.output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            cv2.drawContours(self.cropped, [contour], 0, (0, 0, 255), 3)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            ratio = round(w / h, 1)

            '''
            if w > 80 or h > 80:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            '''
            color = (255, 255, 255)

            # rubik square
            # if 20000 < area < 27000:
            if 200 < area < 5000:
                self.data.append([area, x, y, w, h])
                cv2.rectangle(self.cropped, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.cropped, '{}'.format(area), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
                cv2.putText(self.cropped, '{}, {} ({})'.format(w, h, ratio), (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

            # print('{}, {} ({})'.format(w, h, ratio))

        if self.debug:
            cv2.imshow('OUT', self.output)
            cv2.imshow('ID', self.cropped)

    def density(self):
        #  points = cv2.findNonZero(result2)
        #  rect = cv2.minAreaRect(points)
        return cv2.countNonZero(self.output)

    def has_sample(self, threshold=300):
        if cv2.countNonZero(self.output) > threshold:
            return True
        else:
            return False


