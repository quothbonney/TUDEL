import cv2
import numpy as np
import sys

bound_map = {
    "PbO2": [
        [8, 50, 20],
        [40, 255, 150],
    ],
    "PEDOT": [
        [90, 50, 20],
         [120, 255, 205],
    ],
    "PbI2": [
        [20, 50, 20],
         [35, 255, 205]
    ],
    "Perovskite": [
        [15, 30, 0],
        [80, 255, 255]
    ]
}




# Convert image to grayscale
def size(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(bound_map[sys.argv[1]][0])
    upper = np.array(bound_map[sys.argv[1]][1])
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

    image = cv2.imread(sys.argv[2])

    a = size(image)
    print(f"Width: {a} mm")

    image1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    b = size(image1)
    print(f"Height: {b} mm")