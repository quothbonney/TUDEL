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

FILEPATH = 'imgs/img1.jpg'
TYPE_STRING = "PbO2"

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
    kernel = np.ones((1, 1), np.uint8)
    erosion = cv2.erode(deposit, kernel, iterations=1)

    return deposit


def threshold(res, image, dep):
    ret, thresh = cv2.threshold(res, 128, 255, 0)
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #if area < 1000 and area > 8:
        if area > 9 and area < 600:
            areas.append(area)

    img_gray = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    pixels = cv2.countNonZero(thresh)

    ratio = sum(areas)/pixels

    ratio_string = "{0:.5f}%".format(ratio * 100)

    image = cv2.putText(image, f"Percent Imperfections: {ratio_string}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Image", image)


def mask_size(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    pixels = cv2.countNonZero(thresh)
    return pixels


if __name__ ==  '__main__':


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
    edges = cv2.Canny(image=img_gray, threshold1=8, threshold2=80)


    cv2.imshow("Deposition", deposit)

    threshold(edges, image, deposit)
    sz = mask_size(img_gray)

    cv2.imshow("test", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()