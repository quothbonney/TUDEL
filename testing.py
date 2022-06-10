import cv2
import numpy as np
import matplotlib.pyplot as plt

FILEPATH = 'imgs/Juston PEDOT/JW-10-u-pedotdepo10.jpg'
TYPE_STRING = "PEDOT"

bound_map = {
    "PbO2": [
        [9, 50, 20],
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



def deposition_mask(res, type_string):
    image = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower = np.array(bound_map[type_string][0])
    upper = np.array(bound_map[type_string][1])

    mask = cv2.inRange(image, lower, upper)
    deposit = cv2.bitwise_and(res, res, mask=mask)

    return deposit


def sobel(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)

    grad_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    mask = cv2.threshold(grad, 122, 255, 0)[1]

    return mask


def edge_mask(img):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)

    img_gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)

    grad_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    mask = cv2.threshold(grad, 122, 255, 0)[1]

    return mask


def mask_size(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    pixels = cv2.countNonZero(thresh)
    return pixels


if __name__ == "__main__":
    image = cv2.imread(FILEPATH)
    result = image.copy()
    deposit = deposition_mask(result, TYPE_STRING)

    img_gray = cv2.cvtColor(deposit, cv2.COLOR_BGR2GRAY)

    sobel = sobel(deposit)

    #cv2.imshow('Sobel Image', grad)


    masked = cv2.bitwise_and(deposit, deposit, mask=sobel)
    #plt.imshow(masked)
    sobel_size = mask_size(masked)
    #plt.show()

    #plt.imshow(deposit)
    deposit_size = mask_size(deposit)
    #plt.show()


    edges = edge_mask(deposit)

    final_mask = sobel - edges

    edge_masked = cv2.bitwise_and(deposit, deposit, mask=final_mask)
    #plt.imshow(edge_masked)
    edge_size = mask_size(edge_masked)
    #plt.show()

    ratio = (sobel_size-edge_size)/deposit_size

    ratio_string = "{0:.5f}%".format(ratio * 100)

    print("Sobel size: " + str(sobel_size))
    print("Deposit size: " + str(deposit_size))
    print("Percent Imperfection: " + ratio_string)

    redImg = np.zeros(image.shape, image.dtype)
    redImg[:, :] = (255, 0, 0)

    final = cv2.bitwise_and(image, redImg, mask=final_mask)
    a = cv2.addWeighted(final, 1, image, 1, 0, image)

    plt.imshow(a)
    plt.title('Analysis of ' + TYPE_STRING)
    plt.text(50, 50, f"Percent imperfection: {ratio_string}")

    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()