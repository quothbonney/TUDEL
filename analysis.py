import cv2
import numpy as np
import matplotlib.pyplot as plt

FILEPATH = 'imgs/Ethan PbO2/15M 5mA.jpg'
TYPE_STRING = "PbO2"

# HSV Ranges for each color
# Refer to  https://i.stack.imgur.com/gyuw4.png for new ranges
bound_map = {
    "PbO2": [
        [0, 0, 0],
        [50, 255, 150],
        [160, 0, 0],
        [255, 255, 150]
    ],
    "PEDOT": [
        [90, 50, 20],
         [120, 255, 205],
    ],
    "PbI2": [
        [20, 50, 20],
         [35, 255, 205]
    ],
}


def deposition_mask(res, type_string):
    image = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    if type_string == 'PbO2':
        lower1 = np.array(bound_map[type_string][0])
        upper1 = np.array(bound_map[type_string][1])

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array(bound_map[type_string][2])
        upper2 = np.array(bound_map[type_string][3])

        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)

        mask = upper_mask + lower_mask
    else:
        # lower boundary RED color range values; Hue (0 - 10)
        lower = np.array(bound_map[type_string][0])
        upper = np.array(bound_map[type_string][1])

        mask = cv2.inRange(image, lower, upper)

        # bitwise_and() clips mask to img
    deposit = cv2.bitwise_and(res, res, mask=mask)

    return deposit


def sobel(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)  # Small Gaussian to remove edge noise

    # Sobel solves for x and y separately and then combines them
    grad_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    mask = cv2.threshold(grad, 122, 255, 0)[1]

    return mask


def edge_mask(img):
    # Erodes img to just the edges of the deposit mask
    kernel = np.ones((7, 7), np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)
    erosion = cv2.erode(gauss, kernel, iterations=1)

    # Same process as in sobel()



    grad_x = cv2.Sobel(erosion, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(erosion, cv2.CV_64F, 0, 1, ksize=5)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    mask = cv2.threshold(grad, 122, 255, 0)[1]

    return mask


def mask_size(mask):
    # Count number of pixels in a mask
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]  # Set all pixels = 1, 0;
    pixels = cv2.countNonZero(thresh)  # Basically just a sum of all pixels
    return pixels


if __name__ == "__main__":
    image = cv2.imread(FILEPATH)
    result = image.copy()

    deposit = deposition_mask(result, TYPE_STRING)

    sobel = sobel(deposit)  # Get sobel mask
    sobel_masked = cv2.bitwise_and(deposit, deposit, mask=sobel)  # Get resultant img from mask

    edges = edge_mask(deposit)  # Get edge mask
    edge_masked = cv2.bitwise_and(deposit, deposit, mask=edges)

    deposit_size = mask_size(deposit)
    final_mask = sobel_masked - edge_masked  # Remove edges from mask
    final_size = mask_size(final_mask)

    dst = cv2.addWeighted(result, 1, final_mask, 0.5, 0)  # Merge image and mask together

    plt.imshow(dst)


    ratio = (final_size)/deposit_size

    ratio_string = "{0:.5f}%".format(ratio * 100)

    print("Sobel size: " + str(final_size))
    print("Deposit size: " + str(deposit_size))
    print("Percent Imperfection: " + ratio_string)


    cv2.imshow('argh', deposit)
    plt.title(f'Analysis of {TYPE_STRING} at {FILEPATH}')
    plt.text(50, 50, f"Percent imperfection: {ratio_string}")

    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()