import cv2
import numpy as np
import sys
from mask import Mask
import analysis


def mask_size(mask):
    # Count number of pixels in a mask
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]  # Set all pixels = 1, 0;
    pixels = cv2.countNonZero(thresh)  # Basically just a sum of all pixels
    return pixels


def main(img, type):
    image = img

    result = image.copy()

    try:
        mask = Mask(result, type)
    except Exception:
        print("Cannot create mask")
        sys.exit()

    deposit = mask.deposition_mask()
    dep_masked = cv2.bitwise_and(result, result, mask=deposit)
    sobel = mask.sobel_mask(dep_masked)  # Get sobel mask
    sobel_masked = cv2.bitwise_and(result, result, mask=sobel)
    edges = mask.edge_sobel_mask(dep_masked)

    final_mask = cv2.threshold(sobel-edges, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    errors_masked = cv2.bitwise_and(result, result, mask=final_mask)


    green = np.zeros(result.shape, np.uint8)
    green[:] = (57, 255, 20)

    green_mask = cv2.bitwise_and(green, green, mask=final_mask)



    final_size = analysis.mask_size(errors_masked)
    deposit_size = analysis.mask_size(dep_masked)
    ratio = (final_size)/deposit_size

    ratio_string = "{0:.5f}%".format(ratio * 100)

    print("Sobel size: " + str(final_size))
    print("Deposit size: " + str(deposit_size))
    print("Percent Imperfection: " + ratio_string)

    dst = cv2.bitwise_or(result, green_mask)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst


def percent_imp(img, type):
    image = img

    result = image.copy()

    try:
        mask = Mask(result, type)
    except Exception:
        print("Cannot create mask")
        sys.exit()

    deposit = mask.deposition_mask()
    dep_masked = cv2.bitwise_and(result, result, mask=deposit)
    sobel = mask.sobel_mask(dep_masked)  # Get sobel mask
    sobel_masked = cv2.bitwise_and(result, result, mask=sobel)
    edges = mask.edge_sobel_mask(dep_masked)

    final_mask = cv2.threshold(sobel-edges, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    errors_masked = cv2.bitwise_and(result, result, mask=final_mask)


    green = np.zeros(result.shape, np.uint8)
    green[:] = (57, 255, 20)

    green_mask = cv2.bitwise_and(green, green, mask=final_mask)



    final_size = analysis.mask_size(errors_masked)
    deposit_size = analysis.mask_size(dep_masked)
    ratio = (final_size)/deposit_size

    ratio_string = "{0:.5f}%".format(ratio * 100)

    return ratio_string