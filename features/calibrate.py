import cv2
import numpy as np
import sys
from tkinter import messagebox

counter = 0

def red_brightness(res) -> float:
    image = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 180, 80])
    upper1 = np.array([10, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([100, 180, 80])
    upper2 = np.array([179, 255, 255])

    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    full_mask = upper_mask + lower_mask

    result = cv2.bitwise_and(res, res, mask=full_mask)

    average_hsv = cv2.mean(image, full_mask)
    return average_hsv[2]


def calibrate(img, value):
    global counter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    b_val = red_brightness(image)
    if b_val < 110:
        messagebox.showerror('TUDEL', "ERROR: Cannot Calibrate Image")
        return image

    while b_val < 165:
        counter += 1
        image = calibrate(image, 2)
        b_val = red_brightness(image)

    return image


def getcounter():
    global counter
    return counter

if __name__ == '__main__':
    try:
        image = cv2.imread(sys.argv[1])
        image = calibrate(image, 5)
    except Exception:
        print("Cannot find filepath" + sys.argv[1])
        sys.exit()

    cv2.imwrite(f"output/{sys.argv[1]}_calibrated.png", image)
    print(f"Wrote {sys.argv[1]}_calibrated.png in /output/ ")
