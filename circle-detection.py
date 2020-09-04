import cv2
#import cv2.cv as cv

image='E:\\images\\certa\\c4.1\\PoC\\20200110\\tissue red 2 crop\\120938160_0011.jpg'
img1 = cv2.imread(image)
img = cv2.imread(image,0)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(thresh, 100, 200)
cv2.imshow('edges',edges)

#cv2.imshow('detected ',gray)
cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 30, param2 = 50, minRadius = 0, maxRadius = 0)

for i in circles[0,:]:
    i[2]=i[2]+4
    cv2.circle(img1,(i[0],i[1]),i[2],(0,255,0),2)

#Code to close Window
cv2.imshow('detected Edge',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()