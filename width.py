import cv2
import numpy as np
import matplotlib.pyplot as plt

FILEPATH = 'imgs/random/5.jpg'
one_mm = 7


image = cv2.imread(FILEPATH)

# Convert image to grayscale
def size(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 50, 100])
    upper = np.array([120, 255, 200])
    mask = cv2.inRange(local, lower, upper)
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    widths = []
    for row in thresh:
        width = cv2.countNonZero(row)
        if width > 20:
            widths.append(width)

    average = sum(widths)/len(widths)

    global one_mm
    return str(round(average/one_mm, 3))

if __name__ == '__main__':
    one_mm = 16 * (2/3)

    image = cv2.imread(FILEPATH)

    a = size(image)
    print(f"Width: {a} mm")

    image1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    b = size(image1)
    print(f"Height: {b} mm")