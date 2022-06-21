import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibrate import calibrate
from mask import Mask
import analysis

FILEPATH = 'imgs/1 Att/img1.jpg'
TYPE_STRING = "PbO2"

def mask_size(mask):
    # Count number of pixels in a mask
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]  # Set all pixels = 1, 0;
    pixels = cv2.countNonZero(thresh)  # Basically just a sum of all pixels
    return pixels


if __name__ == "__main__":
    image = cv2.imread(FILEPATH)

    result = image.copy()

    mask = Mask(result, TYPE_STRING)

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
    dst = cv2.bitwise_or(result, green_mask)


    final_size = analysis.mask_size(errors_masked)
    deposit_size = analysis.mask_size(dep_masked)
    ratio = (final_size)/deposit_size

    ratio_string = "{0:.5f}%".format(ratio * 100)

    print("Sobel size: " + str(final_size))
    print("Deposit size: " + str(deposit_size))
    print("Percent Imperfection: " + ratio_string)

    analysis.saturation_histogram(dep_masked)
    analysis.saturation_histogram(sobel_masked)


    cv2.imshow('mask', dep_masked)

    plt.imshow(dst)
    plt.title(f'Analysis of {TYPE_STRING} at {FILEPATH}')
    plt.text(50, 50, f"Percent imperfection: {ratio_string}")
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()