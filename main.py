import cv2
import numpy as np
import sys

bound_map = {
    "PbO2": [
        [10, 50, 20],
         [20, 255, 205]
    ],
    "PEDOT": [
        [90, 50, 20],
         [120, 255, 205]
    ],
    "PbI2": [
        [20, 50, 20],
         [35, 255, 205]
    ],
}


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


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def deposition_mask(res, type_string):
    image = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower = np.array(bound_map[type_string][0])
    upper = np.array(bound_map[type_string][1])

    mask = cv2.inRange(image, lower, upper)

    deposit = cv2.bitwise_and(res, res, mask=mask)
    kernel = np.ones((2, 2))
    erosion = cv2.erode(deposit, kernel, iterations=1)

    cv2.imshow("Deposit", erosion)
    return erosion


def threshold(res, image):
    ret, thresh = cv2.threshold(res, 128, 255, 0)
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    areas = []
    for cnt in contours:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area < 20 and area > 8:
            areas.append(area)

    cv2.imshow("Image", image)
    print(sum(areas))

if __name__ ==  '__main__':
    FILEPATH = 'imgs/img6 PbI2.jpg'
    TYPE_STRING = "PbI2"

    image = cv2.imread(FILEPATH)
    result = image.copy()
    b_val = red_brightness(result)
    if b_val < 110:
        s = input("The image is too dark. It cannot be calibrated without artifacts. "
              "Continue anyways? (y/n) ")
        if s == "n": sys.exit()
        elif s == "y": pass
        else:
            print(s + " is not recognized.")
            sys.exit()

    while b_val < 180:
        result = increase_brightness(result, 5)
        b_val = red_brightness(result)

    #cv2.imshow("final", result)
    deposit = deposition_mask(result, TYPE_STRING)
    img_gray = cv2.cvtColor(deposit, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=img_gray, threshold1=30, threshold2=50)

    threshold(edges, image)



    cv2.waitKey(0)
    cv2.destroyAllWindows()