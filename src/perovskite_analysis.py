import numpy as np
import cv2
from features import analysis
import os


directory = 'imgs/perovskite'


def deposition_mask(image):
    local = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(perovskite[0])
    upper = np.array(perovskite[1])
    mask = cv2.inRange(local, lower, upper)

    return mask

if __name__ == '__main__':
    for FILEPATH in os.listdir(directory):
        print(FILEPATH)
        perovskite = [
            [15, 30, 0],
            [80, 255, 255]
        ]

        image = cv2.imread("imgs/perovskite/" + FILEPATH)
        deposit = deposition_mask(image)

        perovskite[1][0] = 40
        perovskite[0][2] = 40
        deposit2 = deposition_mask(image)

        xor = cv2.bitwise_xor(deposit, deposit2)

        dep_masked = cv2.bitwise_and(image, image, mask=xor)

        green = np.zeros(image.shape, np.uint8)
        green[:] = (57, 255, 20)

        green_mask = cv2.bitwise_and(green, green, mask=deposit2)
        dst = cv2.bitwise_or(image, green_mask)

        cv2.imwrite(FILEPATH + '_analysis.jpg', dst)

        deposit = cv2.threshold(deposit, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        deposit2 = cv2.threshold(deposit2, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

        deposit_masked = cv2.bitwise_and(image, image, mask=deposit)
        errors_masked = cv2.bitwise_and(image, image, mask=deposit2)

        full_size = analysis.mask_size(deposit_masked)
        error_size = analysis.mask_size(errors_masked)

        ratio = (error_size) / full_size

        ratio_string = "{0:.5f}%".format(ratio * 100)

        print(ratio_string)

    cv2.waitKey(0)
    cv2.destroyAllWindows()