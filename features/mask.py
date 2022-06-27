import cv2
import numpy as np
import json

f = open("src/spectrum.json")
bound_map = json.load(f)

class Mask:
    def __init__(self, image, type_string):
        self.image = image
        self.type = type_string

    def deposition_mask(self):
        local = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower = np.array(bound_map[self.type][0])
        upper = np.array(bound_map[self.type][1])
        mask = cv2.inRange(local, lower, upper)

        return mask

    def sobel_mask(self, deposition):
        img_gray = cv2.cvtColor(deposition, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)  # Small Gaussian to remove edge noise

        # Sobel solves for x and y separately and then combines them
        grad_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        mask = cv2.threshold(grad, 122, 255, 0)[1]

        return mask

    def edge_sobel_mask(self, deposition):
        # Erodes img to just the edges of the deposit mask
        kernel = np.ones((5, 5), np.uint8)
        img_gray = cv2.cvtColor(deposition, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)
        erosion = cv2.erode(gauss, kernel, iterations=1)
        dilate = cv2.dilate(erosion, kernel, iterations=1)

        # Same process as in sobel()

        grad_x = cv2.Sobel(dilate, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(dilate, cv2.CV_64F, 0, 1, ksize=5)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        mask = cv2.threshold(grad, 120, 255, 0)[1]

        return mask

f.close()
