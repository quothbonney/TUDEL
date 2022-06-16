import cv2
import numpy as np

FILEPATH = 'imgs/1 Att/carbon.jpg'
one_mm = 7

image = cv2.imread(FILEPATH)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
local = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0])
upper = np.array([255, 255, 50])
mask = cv2.inRange(local, lower, upper)
thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

widths = []
for row in thresh:
    width = cv2.countNonZero(row)
    if width > 80:
        widths.append(width)

average = sum(widths)/len(widths)

print(str(round(average/one_mm, 3)) + " mm")
