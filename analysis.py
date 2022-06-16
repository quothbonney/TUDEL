import numpy as np
import cv2
import matplotlib.pyplot as plt


def mask_size(mask):
    # Count number of pixels in a mask
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]  # Set all pixels = 1, 0;
    pixels = cv2.countNonZero(thresh)  # Basically just a sum of all pixels
    return pixels

def saturation_histogram(image, hsvize=True):
    if hsvize is True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    x = image.flatten()

    filter = x > 50

    fil = x[filter]

    counts, bins = np.histogram(fil, density=True)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel('Saturation')
    plt.ylabel('Probability')
    plt.show()